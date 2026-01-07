"""
Redis-based caching system with exact and semantic search capabilities.

This module provides a two-tier caching strategy:
1. Exact match cache (Hash-based) - O(1) lookup, ~0.1ms
2. Semantic search cache (Vector Similarity Search) - O(log n), ~1-5ms
"""

import re
import hashlib
import pickle
import time
import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import redis
from redis.commands.search.field import VectorField, TextField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    Optimized Redis-based cache manager with dual-tier caching strategy.
    
    Cache Levels:
        1. Exact match (Hash) - O(1), ~0.1ms
        2. Semantic search (VSS) - O(log n), ~1-5ms
    
    Attributes:
        redis_client: Redis client instance
        ttl: Time-to-live for cached entries in seconds
        semantic_threshold: Minimum similarity score for semantic matches
        embedding_dim: Dimension of embedding vectors (BGE-M3 default: 1024)
    """
    
    # Cache key prefixes
    EXACT_PREFIX: str = "cache:exact:"
    SEMANTIC_PREFIX: str = "cache:semantic:"
    STATS_KEY: str = "cache:stats"
    
    def __init__(
        self,
        redis_host: str = "redis",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        ttl: int = 3600,
        semantic_threshold: float = 0.85,
        embedding_dim: int = 1024  # BGE-M3
    ):
        """
        Initialize Redis cache manager with connection and index setup.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis authentication password
            ttl: Cache entry time-to-live in seconds
            semantic_threshold: Minimum cosine similarity for semantic matches
            embedding_dim: Vector embedding dimension size
            
        Raises:
            redis.ConnectionError: If Redis connection fails
        """
        # Establish Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            
            self.redis_client.ping()
            logger.info(f"✓ Redis connected: {redis_host}:{redis_port}")
            
        except redis.ConnectionError as e:
            logger.error(f"✗ Redis connection failed: {e}")
            raise
        
        self.ttl = ttl
        self.semantic_threshold = semantic_threshold
        self.embedding_dim = embedding_dim
        
        # Initialize cache infrastructure
        self._init_vector_index()
        self._init_stats()
    
    def _init_vector_index(self) -> None:
        """
        Create vector similarity search index for semantic caching.
        
        Sets up a Redis Search index with vector field for efficient
        nearest neighbor lookups using cosine similarity.
        """
        try:
            # Check if index already exists
            try:
                self.redis_client.ft("semantic_idx").info()
                logger.info("✓ Vector index already exists")
                return
            except:
                pass
            
            # Create new vector search index
            schema = (
                TextField("query"),
                TextField("response"),
                TagField("language"),
                NumericField("timestamp"),
                NumericField("hit_count"),
                VectorField(
                    "embedding",
                    "FLAT",  # Algorithm: FLAT for exact search, can upgrade to HNSW
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dim,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )
            
            definition = IndexDefinition(
                prefix=[self.SEMANTIC_PREFIX],
                index_type=IndexType.HASH
            )
            
            self.redis_client.ft("semantic_idx").create_index(
                fields=schema,
                definition=definition
            )
            
            logger.info("✓ Vector search index created successfully")
            
        except Exception as e:
            logger.warning(f"Vector index setup warning: {e}")
    
    def _init_stats(self) -> None:
        """Initialize cache statistics tracking if not exists."""
        if not self.redis_client.exists(self.STATS_KEY):
            stats = {
                'hits': 0,
                'misses': 0,
                'semantic_hits': 0,
                'semantic_misses': 0
            }
            self.redis_client.hset(self.STATS_KEY, mapping=stats)
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize query string for consistent cache key generation.
        
        Process:
            1. Remove punctuation
            2. Convert to lowercase
            3. Collapse whitespace
            4. Trim edges
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        normalized = re.sub(r'[^\w\s]', '', query)
        normalized = normalized.lower()
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _generate_cache_key(self, query: str, language: str) -> str:
        """
        Generate MD5 hash-based cache key from normalized query.
        
        Args:
            query: Query string
            language: Language code
            
        Returns:
            32-character hexadecimal cache key
        """
        normalized = self.normalize_query(query)
        cache_string = f"{language}:{normalized}"
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    
    def get_cached_response(
        self, 
        query: str, 
        language: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Retrieve cached response using two-tier lookup strategy.
        
        Lookup Strategy:
            1. Exact match lookup (O(1)) - fastest
            2. Semantic similarity search (O(log n)) - if embedding provided
        
        Args:
            query: Query string to lookup
            language: Language code for the query
            query_embedding: Optional vector embedding for semantic search
            
        Returns:
            Cached response string if found, None otherwise
        """
        try:
            # Tier 1: Exact match lookup
            cache_key = self._generate_cache_key(query, language)
            exact_key = f"{self.EXACT_PREFIX}{cache_key}"
            
            cached = self.redis_client.get(exact_key)
            if cached:
                self.redis_client.hincrby(self.STATS_KEY, 'hits', 1)
                logger.info(f"✓ Exact cache HIT: {query[:50]}")
                return cached.decode('utf-8')
            
            # Tier 2: Semantic similarity search
            if query_embedding is not None:
                semantic_result = self._semantic_search(query_embedding, language)
                
                if semantic_result:
                    cache_key, response, similarity = semantic_result
                    
                    # Increment hit counter for this semantic cache entry
                    self.redis_client.hincrby(
                        f"{self.SEMANTIC_PREFIX}{cache_key}",
                        "hit_count",
                        1
                    )
                    
                    self.redis_client.hincrby(self.STATS_KEY, 'semantic_hits', 1)
                    logger.info(f"✓ Semantic cache HIT (similarity: {similarity:.3f}): {query[:50]}")
                    return response
                else:
                    self.redis_client.hincrby(self.STATS_KEY, 'semantic_misses', 1)
            
            # Cache miss - no match found
            self.redis_client.hincrby(self.STATS_KEY, 'misses', 1)
            logger.info(f"✗ Cache MISS: {query[:50]}")
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    def _semantic_search(
        self, 
        query_embedding: np.ndarray, 
        language: str
    ) -> Optional[Tuple[str, str, float]]:
        """
        Perform semantic similarity search using Redis vector search.
        
        Args:
            query_embedding: Query vector embedding
            language: Language filter for search
            
        Returns:
            Tuple of (cache_key, response, similarity_score) if match found,
            None otherwise
        """
        try:
            embedding_bytes = query_embedding.astype(np.float32).tobytes()
            
            # K-nearest neighbors query
            query = (
                Query(f"@language:{{{language}}} => [KNN 5 @embedding $vec AS score]")
                .return_fields("query", "response", "score")
                .sort_by("score")
                .dialect(2)
            )
            
            results = self.redis_client.ft("semantic_idx").search(
                query,
                query_params={"vec": embedding_bytes}
            )
            
            # Find first result above similarity threshold
            for doc in results.docs:
                distance = float(doc.score)
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= self.semantic_threshold:
                    response = doc.response
                    cache_key = doc.id.replace(self.SEMANTIC_PREFIX, "")
                    return cache_key, response, similarity
            
            return None
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return None
    
    def cache_response(
        self, 
        query: str, 
        response: str, 
        language: str = "az",
        query_embedding: Optional[np.ndarray] = None
    ) -> None:
        """
        Cache response using dual-tier strategy.
        
        Caching Strategy:
            1. Always store in exact match cache
            2. If embedding provided, also store in semantic cache
        
        Args:
            query: Query string
            response: Response to cache
            language: Language code
            query_embedding: Optional vector embedding for semantic caching
        """
        try:
            cache_key = self._generate_cache_key(query, language)
            timestamp = time.time()
            
            # Tier 1: Store in exact match cache
            exact_key = f"{self.EXACT_PREFIX}{cache_key}"
            self.redis_client.setex(
                exact_key,
                self.ttl,
                response.encode('utf-8')
            )
            
            # Tier 2: Store in semantic cache (if embedding provided)
            if query_embedding is not None:
                semantic_key = f"{self.SEMANTIC_PREFIX}{cache_key}"
                embedding_bytes = query_embedding.astype(np.float32).tobytes()
                
                mapping = {
                    "query": query,
                    "response": response,
                    "embedding": embedding_bytes,
                    "language": language,
                    "timestamp": timestamp,
                    "hit_count": 0
                }
                
                pipeline = self.redis_client.pipeline()
                pipeline.hset(semantic_key, mapping=mapping)
                pipeline.expire(semantic_key, self.ttl)
                pipeline.execute()
                
                logger.info(f"✓ Response cached (exact + semantic): {query[:50]}")
            else:
                logger.info(f"✓ Response cached (exact only): {query[:50]}")
                
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def clear_cache(self) -> None:
        """
        Clear all cache entries and reset statistics.
        
        Removes all keys with exact and semantic prefixes,
        then reinitializes the statistics tracking.
        """
        try:
            for prefix in [self.EXACT_PREFIX, self.SEMANTIC_PREFIX]:
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor=cursor,
                        match=f"{prefix}*",
                        count=100
                    )
                    if keys:
                        self.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            
            self.redis_client.delete(self.STATS_KEY)
            self._init_stats()
            
            logger.info("✓ Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive cache statistics and performance metrics.
        
        Returns:
            Dictionary containing:
                - Cache sizes (exact, semantic, total)
                - Hit/miss counts and rates
                - Performance metrics
                - Configuration parameters
        """
        try:
            stats = self.redis_client.hgetall(self.STATS_KEY)
            
            # Extract raw statistics
            hits = int(stats.get(b'hits', 0))
            misses = int(stats.get(b'misses', 0))
            semantic_hits = int(stats.get(b'semantic_hits', 0))
            semantic_misses = int(stats.get(b'semantic_misses', 0))
            
            # Calculate exact cache metrics
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate semantic cache metrics
            semantic_total = semantic_hits + semantic_misses
            semantic_hit_rate = (
                (semantic_hits / semantic_total * 100) 
                if semantic_total > 0 else 0
            )
            
            # Calculate combined metrics
            total_hits = hits + semantic_hits
            combined_rate = (
                (total_hits / (total_requests + semantic_total) * 100)
                if (total_requests + semantic_total) > 0 else 0
            )
            
            # Calculate cache sizes
            exact_size = len(list(self.redis_client.scan_iter(
                match=f"{self.EXACT_PREFIX}*", 
                count=1000
            )))
            semantic_size = len(list(self.redis_client.scan_iter(
                match=f"{self.SEMANTIC_PREFIX}*", 
                count=1000
            )))
            
            return {
                "backend": "redis",
                "exact_cache_size": exact_size,
                "semantic_cache_size": semantic_size,
                "total_cached_items": exact_size + semantic_size,
                
                "exact_hits": hits,
                "exact_misses": misses,
                "exact_hit_rate": round(hit_rate, 2),
                
                "semantic_hits": semantic_hits,
                "semantic_misses": semantic_misses,
                "semantic_hit_rate": round(semantic_hit_rate, 2),
                
                "combined_hit_rate": round(combined_rate, 2),
                "total_cache_hits": total_hits,
                "total_requests": total_requests + semantic_total,
                
                "ttl_seconds": self.ttl,
                "semantic_threshold": self.semantic_threshold,
                "estimated_time_saved_seconds": total_hits * 3
            }
            
        except Exception as e:
            logger.error(f"Statistics retrieval error: {e}")
            return {"error": str(e)}