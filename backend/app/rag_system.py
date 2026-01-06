import os
import yaml
import json
import re
import time
import logging
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import google.generativeai as genai
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import asyncio

from app.cache_manager import RedisCacheManager
from app.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Структура ответа RAG системы"""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    retriever_time: float = 0.0
    llm_time: float = 0.0
    had_subquestions: bool = False
    subquestions: Optional[List[str]] = None
    requires_context: bool = False   
    context_reference: str = ""       
    top_chunks: Optional[List[Dict]] = None
    avg_similarity: float = 0.0
    was_cached: bool = False
    was_semantic_cached: bool = False
    semantic_similarity: float = 0.0
    had_retry: bool = False
    retry_count: int = 0
    llm_tokens: int = 0
    llm_cost: float = 0.0

class LLMCacheManager:
    """Кеш для LLM промптов"""
    
    def __init__(self):
        self.prompt_cache = {}
        self.ttl = 1800  # 30 минут
    
    def get_cached_llm_response(
        self, 
        query: str, 
        context: str, 
        language: str
    ) -> Optional[str]:
        """Кеш для точно таких же контекст+вопрос"""
        cache_key = hashlib.md5(
            f"{query}:{context[:500]}:{language}".encode()
        ).hexdigest()
        
        if cache_key in self.prompt_cache:
            item = self.prompt_cache[cache_key]
            if time.time() - item['timestamp'] < self.ttl:
                logger.info(f"LLM Prompt cache HIT")
                return item['response']
        
        return None
    
    def cache_llm_response(
        self, 
        query: str, 
        context: str, 
        language: str, 
        response: str
    ):
        cache_key = hashlib.md5(
            f"{query}:{context[:500]}:{language}".encode()
        ).hexdigest()
        
        self.prompt_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }

class ABBRAGSystem:
    """RAG система для ABB Bank с оптимизированным семантическим кешем"""
    
    def __init__(
        self,
        gemini_api_key: str,
        openai_api_key: str,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        collection_name: str = "ABB_Knowledge",
        config_path: str = "config/prompts.yaml",
        semantic_cache_threshold: float = 0.85
    ):
        
        # Загрузка конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Инициализация Gemini
        genai.configure(api_key=gemini_api_key)
        client = OpenAI(api_key=openai_api_key)
        
        # LLM модель
        llm_config = self.config['models']['gpt_llm']
        self.llm_model = client
        self.model_params = {
            "model": llm_config['model_name'],
            "temperature": llm_config['temperature'],
            "top_p": llm_config['top_p'],
            "max_tokens": llm_config['max_output_tokens'],
            "n": 1,
        }
        
        # LangChain для chat history
        self.llm_chat = ChatOpenAI(
            model=llm_config['model_name'],
            openai_api_key=openai_api_key,
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_output_tokens']
        )
        
        # Модель для транскрибации
        transcription_config = self.config['models']['gemini_transcription']
        self.transcription_model = genai.GenerativeModel(
            model_name=transcription_config['model_name'],
            generation_config={"temperature": transcription_config['temperature']}
        )
        
        # Milvus подключение
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.collection = Collection(collection_name)
        self.collection.load()
        
        # BGE-M3 embeddings
        self.bge_m3_ef = BGEM3EmbeddingFunction(
            model_name='BAAI/bge-m3',
            device='cpu',
            use_fp16=False
        )
        
        # Кеш-менеджер с семантическим порогом
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        try:
            self.cache_manager = RedisCacheManager(
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=redis_db,
                redis_password=redis_password,
                ttl=3600,
                semantic_threshold=semantic_cache_threshold,
                embedding_dim=1024  # BGE-M3
            )
            logger.info("✓ Redis cache initialized")
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            logger.warning("Falling back to in-memory cache")
            
            # Fallback к старому CacheManager
            from app.cache_manager import CacheManager
            self.cache_manager = CacheManager(
                ttl=3600,
                semantic_threshold=semantic_cache_threshold
            )
        
        # НЕ устанавливаем embedding функцию - будем передавать готовые embeddings
        
        self.llm_cache = LLMCacheManager()
        self.rate_limiter = RateLimiter(
            llm_limit_seconds=int(os.getenv("LLM_RATE_LIMIT_SECONDS", 20)),
            retriever_limit_seconds=int(os.getenv("RETRIEVER_RATE_LIMIT_SECONDS", 10))
        )
        
        # Токенайзер
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Конфигурация
        self.min_similarity = float(os.getenv("MIN_SIMILARITY_THRESHOLD", 0.30))
        self.top_k = int(os.getenv("TOP_K_RESULTS", 3))
        self.cost_per_million = float(os.getenv("COST_PER_MILLION_TOKENS", 15))
        
        # Оптимизации
        self.max_context_tokens = 2000
        self.max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", 20))
        
        logger.info("✓ RAG System initialized with optimized semantic caching")
    
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов"""
        return len(self.tokenizer.encode(text))
    
    def trim_context(self, context: str, max_tokens: int = 2000) -> str:
        """Обрезка контекста до max_tokens"""
        tokens = self.tokenizer.encode(context)
        if len(tokens) <= max_tokens:
            return context
        
        trimmed_tokens = tokens[:max_tokens]
        trimmed_text = self.tokenizer.decode(trimmed_tokens)
        logger.info(f"Context trimmed: {len(tokens)} -> {len(trimmed_tokens)} tokens")
        return trimmed_text
    
    def check_fraud(self, query: str, language: str = "ru") -> Tuple[bool, str]:
        """Проверка на фрод/prompt injection"""
        fraud_prompt = self.config['languages'][language]['fraud_check'].format(query=query)
        
        try:
            response = self.llm_model.chat.completions.create(
                **self.model_params,
                messages=[{"role": "user", "content": fraud_prompt}]
            )
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('is_safe', True), result.get('reason', '')
            
            return True, ""
        except Exception as e:
            logger.error(f"Fraud check error: {e}")
            return True, ""
    
    def format_chat_history_for_decompose(self, chat_history: List) -> str:
        """Форматирование истории чата для decompose_query"""
        if not chat_history:
            return "Нет предыдущего контекста"
        
        trimmed_history = self.trim_chat_history(chat_history)
        
        formatted = []
        for msg in trimmed_history:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Пользователь: {msg.content[:200]}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Ассистент: {msg.content[:200]}")
        
        return "\n".join(formatted)

    def decompose_query(self, query: str, language: str = "ru", chat_history: List = None) -> Dict:
        """Разбиение сложного запроса с учетом истории"""
        history_text = self.format_chat_history_for_decompose(chat_history) if chat_history else "Нет предыдущего контекста"
        
        decompose_prompt = self.config['languages'][language]['decompose_query'].format(
            query=query,
            chat_history=history_text
        )
        
        try:
            response = self.llm_model.chat.completions.create(
                **self.model_params,
                messages=[{"role": "user", "content": decompose_prompt}]
            )
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            
            return {
                "has_subquestions": False,
                "subquestions": [],
                "main_intent": query,
                "requires_context": False,
                "context_reference": ""
            }
        except Exception as e:
            logger.error(f"Query decomposition error: {e}")
            return {
                "has_subquestions": False,
                "subquestions": [],
                "main_intent": query,
                "requires_context": False,
                "context_reference": ""
            }
    
    async def process_subquestions_parallel(
        self, 
        subquestions: List[str]
    ) -> List[Tuple[str, List[Dict], float, np.ndarray]]:
        """Параллельная обработка подвопросов с возвратом embeddings"""
        
        async def search_one(subq):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.search_context, subq)
        
        tasks = [search_one(subq) for subq in subquestions]
        results = await asyncio.gather(*tasks)
        
        return results
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def search_context(
        self, 
        query: str, 
        limit: int = None
    ) -> Tuple[str, List[Dict], float, np.ndarray]:
        """
        Поиск контекста в Milvus с возвратом embedding
        
        Returns:
            (context_text, top_chunks, avg_similarity, query_embedding)
        """
        if limit is None:
            limit = self.top_k
        
        try:
            # Генерация embeddings (делаем один раз!)
            query_embeddings = self.bge_m3_ef.encode_queries([query])
            dense_embedding = query_embeddings['dense'][0]  # Сохраняем для кеша
            
            # Hybrid search запросы
            dense_req = AnnSearchRequest(
                data=[dense_embedding.tolist()],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"ef": 64}},
                limit=limit
            )
            
            sparse_req = AnnSearchRequest(
                data=[query_embeddings['sparse'][[0]]],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=limit
            )
            
            # Выполнение гибридного поиска
            results = self.collection.hybrid_search(
                reqs=[dense_req, sparse_req],
                rerank=WeightedRanker(0.7, 0.3),
                limit=limit,
                output_fields=["text"]
            )
            
            # Обработка результатов
            chunks_with_scores = []
            context_parts = []
            
            for hit in results[0]:
                similarity = hit.score
                text = hit.fields.get('text', '')
                
                if similarity >= self.min_similarity:
                    chunks_with_scores.append({
                        "text": text,
                        "similarity": float(similarity)
                    })
                    context_parts.append(text)
            
            context = "\n\n".join(context_parts)
            avg_similarity = sum(c['similarity'] for c in chunks_with_scores) / len(chunks_with_scores) if chunks_with_scores else 0.0
            
            logger.info(f"Retrieved {len(chunks_with_scores)} chunks, avg similarity: {avg_similarity:.2f}")
            
            # Возвращаем также embedding!
            return context, chunks_with_scores, avg_similarity, dense_embedding
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def trim_chat_history(self, messages: List) -> List:
        """Триминг истории чата"""
        max_messages = int(os.getenv("MAX_HISTORY_MESSAGES", 5))
        first_n = int(os.getenv("HISTORY_FIRST_N", 2))
        last_n = int(os.getenv("HISTORY_LAST_N", 3))
        
        if len(messages) <= max_messages:
            return messages
        
        trimmed = messages[:first_n] + messages[-last_n:]
        logger.info(f"Trimmed chat history: {len(messages)} -> {len(trimmed)} messages")
        
        return trimmed
    
    async def generate_answer_streaming(
        self,
        query: str,
        context: str,
        language: str = "ru",
        chat_history: List = None
    ):
        """Генерация ответа со стримингом"""
        system_prompt = self.config['languages'][language]['system_prompt']
        trimmed_context = self.trim_context(context, max_tokens=self.max_context_tokens)
        
        if chat_history:
            trimmed_history = self.trim_chat_history(chat_history)
            messages = [SystemMessage(content=system_prompt)]
            
            for msg in trimmed_history[-4:]:
                if isinstance(msg, HumanMessage):
                    messages.append(HumanMessage(content=msg.content[:300]))
                elif isinstance(msg, AIMessage):
                    messages.append(AIMessage(content=msg.content[:300]))
            
            messages.append(HumanMessage(
                content=f"КОНТЕКСТ:\n{trimmed_context}\n\nВОПРОС:\n{query}\n\nКраткий ответ:"
            ))
            
            async for chunk in self.llm_chat.astream(messages):
                yield chunk.content
        else:
            prompt = f"{system_prompt}\n\nКОНТЕКСТ:\n{trimmed_context}\n\nВОПРОС:\n{query}\n\nКраткий ответ:"
            
            response = self.llm_model.chat.completions.create(
                **self.model_params,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def generate_answer(
        self,
        query: str,
        context: str,
        language: str = "ru",
        chat_history: List = None
    ) -> str:
        """Генерация ответа с retry логикой"""
        
        # Проверка LLM кеша
        cached = self.llm_cache.get_cached_llm_response(query, context, language)
        if cached:
            return cached
        
        system_prompt = self.config['languages'][language]['system_prompt']
        trimmed_context = self.trim_context(context, max_tokens=self.max_context_tokens)
        
        if chat_history:
            trimmed_history = self.trim_chat_history(chat_history)
            messages = [SystemMessage(content=system_prompt)]
            
            for msg in trimmed_history[-4:]:
                if isinstance(msg, HumanMessage):
                    messages.append(HumanMessage(content=msg.content[:300]))
                elif isinstance(msg, AIMessage):
                    messages.append(AIMessage(content=msg.content[:300]))
            
            messages.append(HumanMessage(
                content=f"КОНТЕКСТ:\n{trimmed_context}\n\nВОПРОС:\n{query}\n\nКраткий ответ:"
            ))
            
            response = self.llm_chat.invoke(messages)
            answer = response.content
        else:
            prompt = f"{system_prompt}\n\nКОНТЕКСТ:\n{trimmed_context}\n\nВОПРОС:\n{query}\n\nКраткий ответ:"
            response = self.llm_model.chat.completions.create(
                **self.model_params,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()
        
        # Кешируем ответ
        self.llm_cache.cache_llm_response(query, context, language, answer)
        
        return answer
    
    async def process_query(
        self,
        query: str,
        user_id: int,
        language: str = "az",
        chat_history: List = None
    ) -> RAGResponse:
        """Основная функция обработки запроса с ОПТИМИЗИРОВАННЫМ семантическим кешем"""
        start_time = time.time()
        retry_count = 0
        had_retry = False
        was_semantic_cached = False
        semantic_similarity = 0.0
        
        # 1. БЫСТРАЯ проверка точного совпадения (БЕЗ embedding!)
        cached_response = self.cache_manager.get_cached_response(
            query, 
            language,
            query_embedding=None  # Не передаем - быстрая проверка!
        )
        
        if cached_response:
            return RAGResponse(
                success=True,
                response=cached_response,
                processing_time=time.time() - start_time,
                was_cached=True,
                was_semantic_cached=False  # Это точное совпадение
            )
        
        # 2. Rate limiting
        can_proceed, limit_msg = self.rate_limiter.check_global_limits(user_id)
        if not can_proceed:
            return RAGResponse(
                success=False,
                error=limit_msg
            )
        
        # 3. Fraud check (только для подозрительных)
        if len(query) > 100 or any(word in query.lower() for word in ['ignore', 'system', 'prompt']):
            is_safe, fraud_reason = self.check_fraud(query, language)
            if not is_safe:
                return RAGResponse(
                    success=False,
                    error=f"Обнаружена потенциальная угроза: {fraud_reason}"
                )
        
        # 4. Chain of Thought
        decomposition = self.decompose_query(query, language, chat_history)
        
        try:
            # 5. Поиск контекста (здесь генерируются embeddings)
            await self.rate_limiter.wait_for_retriever(user_id)
            retriever_start = time.time()
            
            query_embedding = None  # Будет получен из search_context
            
            if decomposition['has_subquestions'] and len(decomposition['subquestions']) <= 3:
                results = await self.process_subquestions_parallel(decomposition['subquestions'])
                
                all_contexts = []
                all_chunks = []
                # Берем embedding ПЕРВОГО подвопроса для кеша
                query_embedding = results[0][3] if results else None
                
                for ctx, chunks, _, _ in results:
                    if ctx:
                        all_contexts.append(ctx)
                        all_chunks.extend(chunks)
                
                context = "\n\n---\n\n".join(all_contexts)
                top_chunks = all_chunks[:self.top_k]
                avg_similarity = sum(c['similarity'] for c in all_chunks) / len(all_chunks) if all_chunks else 0.0
            else:
                # Получаем context И embedding одновременно
                context, top_chunks, avg_similarity, query_embedding = self.search_context(query)
            
            retriever_time = time.time() - retriever_start
            
            # 6. ТЕПЕРЬ проверяем семантический кеш (с уже готовым embedding!)
            if query_embedding is not None:
                semantic_cached = self.cache_manager.get_cached_response(
                    query,
                    language,
                    query_embedding=query_embedding
                )
                
                if semantic_cached:
                    stats = self.cache_manager.get_cache_stats()
                    logger.info("Semantic cache HIT after retrieval!")
                    
                    return RAGResponse(
                        success=True,
                        response=semantic_cached,
                        processing_time=time.time() - start_time,
                        retriever_time=retriever_time,
                        was_cached=True,
                        was_semantic_cached=True,
                        top_chunks=top_chunks,
                        avg_similarity=avg_similarity
                    )
            
            # 7. Проверка наличия контекста
            if not context:
                no_context_msg = self.config['languages'][language]['no_context_response']
                return RAGResponse(
                    success=True,
                    response=no_context_msg,
                    processing_time=time.time() - start_time,
                    retriever_time=retriever_time,
                    had_subquestions=decomposition['has_subquestions'],
                    subquestions=decomposition.get('subquestions', []),
                    requires_context=decomposition.get('requires_context', False),
                    context_reference=decomposition.get('context_reference', '')
                )
            
            # 8. Генерация ответа
            await self.rate_limiter.wait_for_llm(user_id)
            llm_start = time.time()
            
            try:
                answer = self.generate_answer(query, context, language, chat_history)
            except Exception as e:
                logger.error(f"LLM generation failed, retrying: {e}")
                had_retry = True
                retry_count += 1
                answer = self.generate_answer(query, context, language, chat_history)
            
            llm_time = time.time() - llm_start
            
            # 9. Подсчет токенов и стоимости
            llm_tokens = self.count_tokens(answer)
            llm_cost = (llm_tokens / 1_000_000) * self.cost_per_million
            
            # 10. Кеширование с embedding (уже есть из поиска!)
            if query_embedding is not None:
                self.cache_manager.cache_response(
                    query, 
                    answer, 
                    language,
                    query_embedding=query_embedding
                )
            else:
                # Fallback - кешируем без семантики
                self.cache_manager.cache_response(query, answer, language)
            
            logger.info(f"Processing completed: retriever={retriever_time:.2f}s, llm={llm_time:.2f}s")
            
            return RAGResponse(
                success=True,
                response=answer,
                processing_time=time.time() - start_time,
                retriever_time=retriever_time,
                llm_time=llm_time,
                had_subquestions=decomposition['has_subquestions'],
                subquestions=decomposition.get('subquestions', []),
                requires_context=decomposition.get('requires_context', False),
                context_reference=decomposition.get('context_reference', ''),
                top_chunks=top_chunks,
                avg_similarity=avg_similarity,
                had_retry=had_retry,
                retry_count=retry_count,
                llm_tokens=llm_tokens,
                llm_cost=llm_cost,
                was_semantic_cached=False
            )
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return RAGResponse(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def transcribe_audio(self, audio_file_path: str) -> str:
        """Транскрибация аудио через Gemini"""
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            file_size = os.path.getsize(audio_file_path)
            logger.info(f"Transcribing audio file: {audio_file_path}, size: {file_size} bytes")
            
            audio_file = genai.upload_file(path=audio_file_path)
            
            max_attempts = 10
            for attempt in range(max_attempts):
                file_status = genai.get_file(audio_file.name)
                if file_status.state.name == "ACTIVE":
                    break
                elif file_status.state.name == "FAILED":
                    raise Exception("Audio file processing failed")
                logger.info(f"Waiting for file to be processed... attempt {attempt + 1}/{max_attempts}")
                await asyncio.sleep(2)
            else:
                raise Exception("Timeout waiting for audio file to be processed")
            
            prompt = """Пожалуйста, транскрибируйте этот аудиофайл точно и полностью.
            Верните только текст речи без дополнительных комментариев.
            Если речь на азербайджанском языке, транскрибируйте на азербайджанском.
            Если на русском - на русском. Если на английском - на английском."""
            
            response = self.transcription_model.generate_content([prompt, audio_file])
            transcription = response.text.strip()
            
            try:
                genai.delete_file(audio_file.name)
            except:
                pass
            
            logger.info(f"Transcription result: {transcription[:100]}...")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise Exception(f"Ошибка транскрибации: {str(e)}")