import time
import asyncio
from typing import Dict, Tuple, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for LLM and Retriever requests.
    
    This class implements rate limiting for two types of operations:
    - LLM requests: Limited by time interval between consecutive requests
    - Retriever requests: Limited by time interval between consecutive searches
    
    Additionally, it enforces global limits per minute and per hour for all operations.
    
    Attributes:
        llm_limit_seconds: Minimum seconds between LLM requests
        retriever_limit_seconds: Minimum seconds between Retriever requests
        max_per_minute: Maximum requests allowed per minute
        max_per_hour: Maximum requests allowed per hour
    """
    
    def __init__(
        self,
        llm_limit_seconds: int = 20,
        retriever_limit_seconds: int = 10,
        max_per_minute: int = 20,
        max_per_hour: int = 100
    ) -> None:
        self.llm_limit_seconds: int = llm_limit_seconds
        self.retriever_limit_seconds: int = retriever_limit_seconds
        self.max_per_minute: int = max_per_minute
        self.max_per_hour: int = max_per_hour
        
        # Storage for tracking last request timestamps per user
        self.llm_last_request: Dict[int, float] = {}
        self.retriever_last_request: Dict[int, float] = {}
        self.minute_requests: Dict[int, List[float]] = defaultdict(list)
        self.hour_requests: Dict[int, List[float]] = defaultdict(list)
    
    async def check_llm_limit(self, user_id: int) -> Tuple[bool, str]:
        """
        Check rate limit for LLM requests.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple containing:
                - bool: True if request is allowed, False otherwise
                - str: Status message or error description
        """
        now = time.time()
        
        if user_id in self.llm_last_request:
            elapsed = now - self.llm_last_request[user_id]
            if elapsed < self.llm_limit_seconds:
                wait_time = self.llm_limit_seconds - elapsed
                return False, f"Please wait {wait_time:.1f} seconds before next request"
        
        self.llm_last_request[user_id] = now
        return True, "OK"
    
    async def check_retriever_limit(self, user_id: int) -> Tuple[bool, str]:
        """
        Check rate limit for Retriever requests.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple containing:
                - bool: True if request is allowed, False otherwise
                - str: Status message or error description
        """
        now = time.time()
        
        if user_id in self.retriever_last_request:
            elapsed = now - self.retriever_last_request[user_id]
            if elapsed < self.retriever_limit_seconds:
                wait_time = self.retriever_limit_seconds - elapsed
                return False, f"Please wait {wait_time:.1f} seconds before next search"
        
        self.retriever_last_request[user_id] = now
        return True, "OK"
    
    def check_global_limits(self, user_id: int) -> Tuple[bool, str]:
        """
        Check global rate limits (per minute and per hour).
        
        This method cleans up expired timestamps and enforces both
        per-minute and per-hour request limits.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple containing:
                - bool: True if request is allowed, False otherwise
                - str: Status message or error description
        """
        now = time.time()
        
        # Clean up expired request timestamps
        self.minute_requests[user_id] = [
            t for t in self.minute_requests[user_id] if now - t < 60
        ]
        self.hour_requests[user_id] = [
            t for t in self.hour_requests[user_id] if now - t < 3600
        ]
        
        if len(self.minute_requests[user_id]) >= self.max_per_minute:
            return False, "Per-minute request limit exceeded"
        
        if len(self.hour_requests[user_id]) >= self.max_per_hour:
            return False, "Per-hour request limit exceeded"
        
        # Register current request
        self.minute_requests[user_id].append(now)
        self.hour_requests[user_id].append(now)
        
        return True, "OK"
    
    async def wait_for_llm(self, user_id: int) -> None:
        """
        Wait until LLM rate limit allows the next request.
        
        This method automatically calculates and waits for the required
        time before the next LLM request can be made.
        
        Args:
            user_id: Unique identifier for the user
        """
        can_proceed, msg = await self.check_llm_limit(user_id)
        if not can_proceed:
            wait_time = self.llm_limit_seconds
            if user_id in self.llm_last_request:
                wait_time = self.llm_limit_seconds - (time.time() - self.llm_last_request[user_id])
            logger.info(f"Waiting {wait_time}s for LLM rate limit")
            await asyncio.sleep(wait_time)
    
    async def wait_for_retriever(self, user_id: int) -> None:
        """
        Wait until Retriever rate limit allows the next request.
        
        This method automatically calculates and waits for the required
        time before the next Retriever request can be made.
        
        Args:
            user_id: Unique identifier for the user
        """
        can_proceed, msg = await self.check_retriever_limit(user_id)
        if not can_proceed:
            wait_time = self.retriever_limit_seconds
            if user_id in self.retriever_last_request:
                wait_time = self.retriever_limit_seconds - (time.time() - self.retriever_last_request[user_id])
            logger.info(f"Waiting {wait_time}s for Retriever rate limit")
            await asyncio.sleep(wait_time)