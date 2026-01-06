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
    Оптимизированный кеш-менеджер с Redis
    
    Уровни кеша:
    1. Exact match (Hash) - O(1), ~0.1ms
    2. Semantic search (VSS) - O(log n), ~1-5ms
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        ttl: int = 3600,
        semantic_threshold: float = 0.85,
        embedding_dim: int = 1024  # BGE-M3
    ):
        # Подключение к Redis
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
        
        # Префиксы
        self.EXACT_PREFIX = "cache:exact:"
        self.SEMANTIC_PREFIX = "cache:semantic:"
        self.STATS_KEY = "cache:stats"
        
        # Инициализация
        self._init_vector_index()
        self._init_stats()
    
    def _init_vector_index(self):
        """Создание VSS индекса для семантического поиска"""
        try:
            # Проверка существования индекса
            try:
                self.redis_client.ft("semantic_idx").info()
                logger.info("✓ Vector index exists")
                return
            except:
                pass
            
            # Создание нового индекса
            schema = (
                TextField("query"),
                TextField("response"),
                TagField("language"),
                NumericField("timestamp"),
                NumericField("hit_count"),
                VectorField(
                    "embedding",
                    "FLAT",  # Для начала, потом можно HNSW
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
            
            logger.info("✓ Vector index created")
            
        except Exception as e:
            logger.warning(f"Vector index setup: {e}")
    
    def _init_stats(self):
        """Инициализация статистики"""
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
        """Нормализация запроса"""
        normalized = re.sub(r'[^\w\s]', '', query)
        normalized = normalized.lower()
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _generate_cache_key(self, query: str, language: str) -> str:
        """Генерация ключа"""
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
        Получение кешированного ответа
        
        Стратегия:
        1. Точное совпадение (O(1))
        2. Семантический поиск (если есть embedding)
        """
        try:
            # 1. Точное совпадение
            cache_key = self._generate_cache_key(query, language)
            exact_key = f"{self.EXACT_PREFIX}{cache_key}"
            
            cached = self.redis_client.get(exact_key)
            if cached:
                self.redis_client.hincrby(self.STATS_KEY, 'hits', 1)
                logger.info(f"✓ Exact cache HIT: {query[:50]}")
                return cached.decode('utf-8')
            
            # 2. Семантический поиск
            if query_embedding is not None:
                logger.info(f"/////////////////////////////////////")
                semantic_result = self._semantic_search(query_embedding, language)
                logger.info(f"/{semantic_result}")
                
                if semantic_result:
                    cache_key, response, similarity = semantic_result
                    
                    # Увеличиваем счетчик
                    self.redis_client.hincrby(
                        f"{self.SEMANTIC_PREFIX}{cache_key}",
                        "hit_count",
                        1
                    )
                    
                    self.redis_client.hincrby(self.STATS_KEY, 'semantic_hits', 1)
                    logger.info(f"✓ Semantic HIT ({similarity:.3f}): {query[:50]}")
                    return response
                else:
                    self.redis_client.hincrby(self.STATS_KEY, 'semantic_misses', 1)
            
            # 3. Cache miss
            self.redis_client.hincrby(self.STATS_KEY, 'misses', 1)
            logger.info(f"✗ Cache MISS: {query[:50]}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def _semantic_search(
        self, 
        query_embedding: np.ndarray, 
        language: str
    ) -> Optional[Tuple[str, str, float]]:
        """Семантический поиск через Redis VSS"""
        try:
            embedding_bytes = query_embedding.astype(np.float32).tobytes()
            
            # KNN запрос
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
            
            for doc in results.docs:
                distance = float(doc.score)
                similarity = 1 - distance
                
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
    ):
        """Кеширование ответа"""
        try:
            cache_key = self._generate_cache_key(query, language)
            timestamp = time.time()
            
            # 1. Точный кеш
            exact_key = f"{self.EXACT_PREFIX}{cache_key}"
            self.redis_client.setex(
                exact_key,
                self.ttl,
                response.encode('utf-8')
            )
            
            # 2. Семантический кеш
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
                
                logger.info(f"✓ Cached (exact+semantic): {query[:50]}")
            else:
                logger.info(f"✓ Cached (exact): {query[:50]}")
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear_cache(self):
        """Очистка кешей"""
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
            
            logger.info("✓ Cache cleared")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кеша"""
        try:
            stats = self.redis_client.hgetall(self.STATS_KEY)
            
            hits = int(stats.get(b'hits', 0))
            misses = int(stats.get(b'misses', 0))
            semantic_hits = int(stats.get(b'semantic_hits', 0))
            semantic_misses = int(stats.get(b'semantic_misses', 0))
            
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            semantic_total = semantic_hits + semantic_misses
            semantic_hit_rate = (
                (semantic_hits / semantic_total * 100) 
                if semantic_total > 0 else 0
            )
            
            total_hits = hits + semantic_hits
            combined_rate = (
                (total_hits / (total_requests + semantic_total) * 100)
                if (total_requests + semantic_total) > 0 else 0
            )
            
            # Размеры
            exact_size = len(list(self.redis_client.scan_iter(match=f"{self.EXACT_PREFIX}*", count=1000)))
            semantic_size = len(list(self.redis_client.scan_iter(match=f"{self.SEMANTIC_PREFIX}*", count=1000)))
            
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
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}