import time
import asyncio
from typing import Dict, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter для LLM и Retriever"""
    
    def __init__(
        self,
        llm_limit_seconds: int = 20,
        retriever_limit_seconds: int = 10,
        max_per_minute: int = 20,
        max_per_hour: int = 100
    ):
        self.llm_limit_seconds = llm_limit_seconds
        self.retriever_limit_seconds = retriever_limit_seconds
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        
        # Хранение последних запросов по пользователям
        self.llm_last_request: Dict[int, float] = {}
        self.retriever_last_request: Dict[int, float] = {}
        self.minute_requests: Dict[int, list] = defaultdict(list)
        self.hour_requests: Dict[int, list] = defaultdict(list)
    
    async def check_llm_limit(self, user_id: int) -> Tuple[bool, str]:
        """Проверка лимита для LLM"""
        now = time.time()
        
        if user_id in self.llm_last_request:
            elapsed = now - self.llm_last_request[user_id]
            if elapsed < self.llm_limit_seconds:
                wait_time = self.llm_limit_seconds - elapsed
                return False, f"Подождите {wait_time:.1f} секунд перед следующим запросом"
        
        self.llm_last_request[user_id] = now
        return True, "OK"
    
    async def check_retriever_limit(self, user_id: int) -> Tuple[bool, str]:
        """Проверка лимита для Retriever"""
        now = time.time()
        
        if user_id in self.retriever_last_request:
            elapsed = now - self.retriever_last_request[user_id]
            if elapsed < self.retriever_limit_seconds:
                wait_time = self.retriever_limit_seconds - elapsed
                return False, f"Подождите {wait_time:.1f} секунд перед следующим поиском"
        
        self.retriever_last_request[user_id] = now
        return True, "OK"
    
    def check_global_limits(self, user_id: int) -> Tuple[bool, str]:
        """Проверка глобальных лимитов (в минуту/час)"""
        now = time.time()
        
        # Очистка старых запросов
        self.minute_requests[user_id] = [
            t for t in self.minute_requests[user_id] if now - t < 60
        ]
        self.hour_requests[user_id] = [
            t for t in self.hour_requests[user_id] if now - t < 3600
        ]
        
        if len(self.minute_requests[user_id]) >= self.max_per_minute:
            return False, "Превышен лимит запросов в минуту"
        
        if len(self.hour_requests[user_id]) >= self.max_per_hour:
            return False, "Превышен лимит запросов в час"
        
        # Добавляем текущий запрос
        self.minute_requests[user_id].append(now)
        self.hour_requests[user_id].append(now)
        
        return True, "OK"
    
    async def wait_for_llm(self, user_id: int):
        """Ожидание перед LLM запросом"""
        can_proceed, msg = await self.check_llm_limit(user_id)
        if not can_proceed:
            wait_time = self.llm_limit_seconds
            if user_id in self.llm_last_request:
                wait_time = self.llm_limit_seconds - (time.time() - self.llm_last_request[user_id])
            logger.info(f"Waiting {wait_time}s for LLM rate limit")
            await asyncio.sleep(wait_time)
    
    async def wait_for_retriever(self, user_id: int):
        """Ожидание перед Retriever запросом"""
        can_proceed, msg = await self.check_retriever_limit(user_id)
        if not can_proceed:
            wait_time = self.retriever_limit_seconds
            if user_id in self.retriever_last_request:
                wait_time = self.retriever_limit_seconds - (time.time() - self.retriever_last_request[user_id])
            logger.info(f"Waiting {wait_time}s for Retriever rate limit")
            await asyncio.sleep(wait_time)