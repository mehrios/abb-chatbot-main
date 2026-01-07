"""
ABB Bank Knowledge Base - Milvus RAG Pipeline
Hybrid search using BGE-M3 (dense + sparse vectors)
Docker-ready version
"""

import os
import json
import sys
import time
import socket
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    model,
    utility,
    AnnSearchRequest,
    WeightedRanker
)
from transformers import AutoTokenizer


class MilvusRAGPipeline:
    """
    Pipeline for indexing and searching in Milvus with hybrid approach.
    
    Combines dense and sparse vectors using BGE-M3 model for optimal
    semantic and keyword-based retrieval.
    
    Attributes:
        collection_name: Name of the Milvus collection
        model_name: HuggingFace model identifier for embeddings
        device: Device for model inference ('cpu' or 'cuda')
        host: Milvus server host
        port: Milvus server port
        tokenizer: Tokenizer for text chunking
        bge_m3_ef: BGE-M3 embedding function
        collection: Milvus collection instance
    """
    
    def __init__(
        self,
        collection_name: str = "ABB_Knowledge",
        host: Optional[str] = None,
        port: Optional[str] = None,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu"
    ) -> None:
        """
        Initialize the RAG pipeline with Milvus connection and embedding model.
        
        Args:
            collection_name: Name for the Milvus collection
            host: Milvus host address (defaults to env var or 'milvus')
            port: Milvus port (defaults to env var or '19530')
            model_name: HuggingFace model for embeddings
            device: Compute device ('cpu' or 'cuda')
        """
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        
        # Get connection parameters from environment variables
        self.host = host or os.getenv("MILVUS_HOST", "milvus")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        
        print(f"🔌 Connecting to Milvus: {self.host}:{self.port}")
        
        # Wait for Milvus availability
        self._wait_for_milvus()
        
        # Initialize tokenizer
        print(f"📦 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize embedding model
        print(f"🧠 Loading embedding model: {model_name}")
        self.bge_m3_ef = model.hybrid.BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=False
        )
        
        # Connect to Milvus
        connections.connect("default", host=self.host, port=self.port)
        
        # Initialize collection
        self.collection = self._init_collection()
    
    def _wait_for_milvus(
        self,
        max_retries: int = 30,
        delay: int = 2
    ) -> None:
        """
        Wait for Milvus to become available.
        
        Args:
            max_retries: Maximum number of connection attempts
            delay: Delay in seconds between attempts
            
        Raises:
            ConnectionError: If Milvus is not available after max retries
        """
        print(f"⏳ Waiting for Milvus readiness ({self.host}:{self.port})...")
        
        for attempt in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((self.host, int(self.port)))
                sock.close()
                
                if result == 0:
                    print(f"✅ Milvus is available!")
                    time.sleep(5)  # Additional wait for full initialization
                    return
                    
            except Exception as e:
                pass
            
            print(f"   Attempt {attempt + 1}/{max_retries}...")
            time.sleep(delay)
        
        raise ConnectionError(
            f"❌ Failed to connect to Milvus after {max_retries * delay} seconds"
        )
        
    def _init_collection(self) -> Collection:
        """
        Create or recreate collection with proper schema.
        
        Schema includes:
        - id: Auto-generated primary key
        - dense_vector: 1024-dim dense embeddings
        - sparse_vector: Sparse keyword-based embeddings
        - text: Actual text content
        - path: Hierarchical path in knowledge base
        - url: Source URL
        
        Returns:
            Initialized Milvus collection
        """
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500)
        ]
        
        schema = CollectionSchema(fields, "ABB Bank Hierarchical Knowledge Base")
        
        if utility.has_collection(self.collection_name):
            print(f"⚠️  Collection '{self.collection_name}' exists. Dropping...")
            try:
                col = Collection(self.collection_name)
                col.release()
                utility.drop_collection(self.collection_name)
                time.sleep(2)
            except Exception as e:
                print(f"Warning during drop: {e}")
        
        print(f"✨ Creating new collection: {self.collection_name}")
        return Collection(name=self.collection_name, schema=schema)
    
    def chunk_text_by_tokens(
        self,
        text: str,
        max_tokens: int = 200,
        overlap: int = 12
    ) -> List[str]:
        """
        Split text into chunks by token count with overlap.
        
        This ensures chunks respect token boundaries rather than character
        boundaries, which is important for transformer models.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: List[str] = []
        step = max_tokens - overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            if i + max_tokens >= len(tokens):
                break
                
        return chunks
    
    def insert_hierarchical_data(
        self,
        data: Dict[str, Any],
        current_path: str = ""
    ) -> int:
        """
        Recursively traverse JSON tree and index text content.
        
        This method walks through the hierarchical knowledge base structure,
        identifies text content, chunks it, generates embeddings, and inserts
        into Milvus collection.
        
        Args:
            data: Hierarchical data structure (nested dicts)
            current_path: Current path in hierarchy (for tracking)
            
        Returns:
            Total number of chunks indexed
        """
        # Metadata keys to skip (not content)
        meta_keys = {
            'url', 'links', 'faq', 'nested_pages', 'scrollable_items',
            'level', 'parent_path', 'timestamp', 'content_hash',
            'title', 'interactive_content'
        }
        
        total_chunks = 0
        
        for key, value in data.items():
            if key in meta_keys:
                continue
            
            new_path = f"{current_path}/{key}" if current_path else key
            
            if isinstance(value, dict):
                content = value.get("text_content", "")
                
                if content and len(content.strip()) > 50:
                    print(f"📥 Processing: {new_path}")
                    
                    chunks = self.chunk_text_by_tokens(content)
                    url_val = value.get("url", "")
                    url = url_val if isinstance(url_val, str) else url_val.get("url", "")
                    
                    outputs = self.bge_m3_ef.encode_documents(chunks)
                    dense_vecs = outputs['dense']
                    sparse_vecs = outputs['sparse']
                    
                    rows: List[Dict[str, Any]] = []
                    for i, chunk in enumerate(chunks):
                        s_vec = sparse_vecs[i]
                        
                        # Handle sparse vectors from scipy.sparse format
                        if hasattr(s_vec, 'tocoo'):
                            # Convert to COO format if not already
                            s_vec_coo = s_vec.tocoo()
                            sparse_dict = {
                                int(idx): float(val) 
                                for idx, val in zip(s_vec_coo.col, s_vec_coo.data)
                            }
                        elif hasattr(s_vec, 'col') and hasattr(s_vec, 'data'):
                            # Already in COO format
                            sparse_dict = {
                                int(idx): float(val) 
                                for idx, val in zip(s_vec.col, s_vec.data)
                            }
                        elif hasattr(s_vec, "to_dict"):
                            sparse_dict = s_vec.to_dict()
                        elif isinstance(s_vec, dict):
                            sparse_dict = s_vec
                        else:
                            # Fallback for other types
                            try:
                                sparse_dict = {int(k): float(v) for k, v in s_vec.items()}
                            except AttributeError:
                                print(f"   ⚠️ Unknown sparse vector format: {type(s_vec)}")
                                sparse_dict = {}

                        rows.append({
                            "dense_vector": dense_vecs[i].tolist(),
                            "sparse_vector": sparse_dict,
                            "text": str(chunk),
                            "path": str(new_path),
                            "url": str(url)
                        })
                    
                    if rows:
                        try:
                            self.collection.insert(rows)
                            total_chunks += len(rows)
                            print(f"   ✅ {len(rows)} chunks indexed")
                        except Exception as e:
                            print(f"   ❌ Insert error in {new_path}: {e}")
                
                # Recursively process nested sections
                sub_sections = {
                    k: v for k, v in value.items()
                    if k not in meta_keys and k != 'text_content' and isinstance(v, dict)
                }
                
                if sub_sections:
                    total_chunks += self.insert_hierarchical_data(sub_sections, new_path)
        
        return total_chunks
    
    def create_indexes(self) -> None:
        """
        Create indexes for optimized search performance.
        
        Creates two indexes:
        1. HNSW index for dense vectors (semantic search)
        2. Sparse inverted index for sparse vectors (keyword search)
        
        Then loads the collection into memory for fast querying.
        """
        print("🔧 Creating index for dense vectors...")
        self.collection.create_index(
            field_name="dense_vector",
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 64}
            }
        )
        
        print("🔧 Creating index for sparse vectors...")
        self.collection.create_index(
            field_name="sparse_vector",
            index_params={
                "metric_type": "IP",
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"drop_ratio_build": 0.2}
            }
        )
        
        print("📊 Loading collection into memory...")
        self.collection.load()
        print("✅ Indexes created and collection loaded!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main pipeline function.
    
    Orchestrates the entire indexing process:
    1. Load JSON data
    2. Initialize Milvus connection
    3. Process and chunk text
    4. Generate embeddings
    5. Index in Milvus
    6. Create search indexes
    """
    # Configuration
    JSON_FILE = "abb_bank_hierarchical_data.json"
    COLLECTION_NAME = "ABB_Knowledge"
    
    print("=" * 60)
    print("🚀 ABB BANK KNOWLEDGE BASE - MILVUS RAG PIPELINE")
    print("=" * 60)
    
    # Check for JSON file
    if not Path(JSON_FILE).exists():
        print(f"❌ File not found: {JSON_FILE}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = MilvusRAGPipeline(collection_name=COLLECTION_NAME)
    
    # Load data
    print(f"\n📖 Loading data from: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Index data
    print("\n🔄 Starting indexing...")
    
    if 'Abb' in full_data:
        total_chunks = pipeline.insert_hierarchical_data(full_data['Abb'])
    else:
        total_chunks = pipeline.insert_hierarchical_data(full_data)
    
    print(f"\n✅ Indexing completed! Total chunks: {total_chunks}")
    
    # Flush buffer and create indexes
    pipeline.collection.flush()
    pipeline.create_indexes()
    
    print("\n" + "=" * 60)
    print("✨ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)