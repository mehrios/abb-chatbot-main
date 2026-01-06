import os
import logging
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from dotenv import load_dotenv
import tempfile
import json

from app.database import get_db, init_db
from app import crud, schemas, models
from app.rag_system import ABBRAGSystem, RAGResponse
from langchain_core.messages import HumanMessage, AIMessage

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title="ABB Bank Chatbot API",
    description="AI-powered chatbot for ABB Bank with RAG system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация RAG системы
_rag_system_instance = None
_rag_lock = None

def get_rag_system() -> ABBRAGSystem:
    """
    Синглтон для RAG системы с персистентным Redis кешем
    """
    global _rag_system_instance
    
    if _rag_system_instance is None:
        logger.info("Initializing RAG system singleton...")
        _rag_system_instance = ABBRAGSystem(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=os.getenv("MILVUS_PORT", "19530"),
            collection_name=os.getenv("MILVUS_COLLECTION_NAME", "ABB_Knowledge")
        )
        logger.info("✓ RAG system singleton initialized with Redis cache")
    
    return _rag_system_instance

# ============= STARTUP/SHUTDOWN =============
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    init_db()
    
    # Инициализация RAG системы
    get_rag_system()
    
    logger.info("✓ Application started with Redis cache")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке"""
    global _rag_system_instance
    
    if _rag_system_instance:
        stats = _rag_system_instance.cache_manager.get_cache_stats()
        logger.info(f"Final cache stats: {stats}")
    
    logger.info("Application shutting down...")

# ============= HEALTH CHECK =============
@app.get("/")
async def root():
    return {"status": "ok", "service": "ABB Bank Chatbot API"}

@app.get("/health")
async def health_check():
    rag = get_rag_system()
    cache_stats = rag.cache_manager.get_cache_stats()
    
    return {
        "status": "healthy",
        "rag_system": "ready",
        "cache": {
            "backend": cache_stats.get("backend"),
            "exact_size": cache_stats.get("exact_cache_size"),
            "semantic_size": cache_stats.get("semantic_cache_size"),
            "hit_rate": f"{cache_stats.get('combined_hit_rate', 0)}%"
        }
    }

# ============= AUTH ENDPOINTS =============
@app.post("/auth/register", response_model=schemas.UserResponse)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Регистрация нового пользователя"""
    existing_user = crud.get_user_by_username(db, user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    return crud.create_user(db, user)

@app.post("/auth/login", response_model=schemas.UserResponse)
async def login(credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    """Вход пользователя"""
    user = crud.verify_user(db, credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    return user

@app.post("/auth/reset-password")
async def reset_password(reset: schemas.PasswordReset, db: Session = Depends(get_db)):
    """Сброс пароля"""
    user = crud.reset_password(db, reset.username, reset.new_password)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "Password reset successfully"}

# ============= CHAT ENDPOINTS =============
@app.post("/chats", response_model=schemas.ChatResponse)
async def create_chat(chat: schemas.ChatCreate, db: Session = Depends(get_db)):
    """Создание нового чата"""
    return crud.create_chat(db, chat)

@app.get("/chats/{chat_id}", response_model=schemas.ChatResponse)
async def get_chat(chat_id: str, db: Session = Depends(get_db)):
    """Получение информации о чате"""
    chat = crud.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@app.get("/users/{user_id}/chats", response_model=List[schemas.ChatResponse])
async def get_user_chats(user_id: int, db: Session = Depends(get_db)):
    """Получение всех чатов пользователя"""
    return crud.get_user_chats(db, user_id)

@app.get("/chats/{chat_id}/messages", response_model=List[schemas.MessageResponse])
async def get_chat_messages(chat_id: str, db: Session = Depends(get_db)):
    """Получение сообщений чата"""
    return crud.get_chat_messages(db, chat_id)

# ============= QUERY ENDPOINTS =============
@app.post("/query", response_model=schemas.QueryResponse)
async def process_query(request: schemas.QueryRequest, db: Session = Depends(get_db)):
    """Обработка текстового запроса"""
    rag_system = get_rag_system()
    try:
        # Получение истории чата
        messages = crud.get_chat_messages(db, request.chat_id)
        chat_history = []
        for msg in messages[-10:]:  # Последние 10 сообщений
            if msg.author == "human":
                chat_history.append(HumanMessage(content=msg.message))
            else:
                chat_history.append(AIMessage(content=msg.message))
        
        # Обработка запроса через RAG
        result = await rag_system.process_query(
            query=request.query,
            user_id=request.user_id,
            language=request.language,
            chat_history=chat_history
        )
        
        if not result.success:
            return schemas.QueryResponse(
                success=False,
                response=result.error,
                error=result.error,
                subquestions=[]
            )
        
        # Сохранение сообщения пользователя
        user_msg = schemas.MessageCreate(
            chat_id=request.chat_id,
            user_id=request.user_id,
            message=request.query,
            author="human",
            token_count=rag_system.count_tokens(request.query)
        )
        crud.create_message(db, user_msg)
        
        # Сохранение ответа бота
        bot_msg = schemas.MessageCreate(
            chat_id=request.chat_id,
            user_id=request.user_id,
            message=result.response,
            author="bot",
            token_count=result.llm_tokens
        )
        crud.create_message(db, bot_msg)
        
        # Логирование запроса
        log_data = {
            "chat_id": request.chat_id,
            "user_id": request.user_id,
            "query": request.query,
            "language": request.language,
            "retriever_time": result.retriever_time,
            "top_k_chunks": result.top_chunks,
            "avg_similarity": result.avg_similarity,
            "retriever_used": True,
            "llm_response": result.response,
            "llm_time": result.llm_time,
            "llm_tokens": result.llm_tokens,
            "llm_cost": result.llm_cost,
            "had_subquestions": result.had_subquestions,
            "subquestions": result.subquestions or [],
            "requires_context": result.requires_context, 
            "context_reference": result.context_reference, 
            "cot_total_time": result.processing_time if result.had_subquestions else None,
            "was_cached": result.was_cached,
            "had_retry": result.had_retry,
            "retry_count": result.retry_count
        }
        query_log = crud.create_query_log(db, log_data)
        
        return schemas.QueryResponse(
            success=True,
            response=result.response,
            processing_time=result.processing_time,
            had_subquestions=result.had_subquestions,
            subquestions=result.subquestions or [],
            requires_context=result.requires_context,   
            context_reference=result.context_reference,  
            was_cached=result.was_cached,
            query_log_id=query_log.id
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def process_query_stream(request: schemas.QueryRequest, db: Session = Depends(get_db)):
    """Обработка запроса со стримингом ответа"""
    rag_system = get_rag_system()
    async def generate():
        try:
            # Получение истории
            messages = crud.get_chat_messages(db, request.chat_id)
            chat_history = []
            for msg in messages[-10:]:
                if msg.author == "human":
                    chat_history.append(HumanMessage(content=msg.message))
                else:
                    chat_history.append(AIMessage(content=msg.message))
            
            # Сначала делаем retrieval
            await rag_system.rate_limiter.wait_for_retriever(request.user_id)
            
            decomposition = rag_system.decompose_query(request.query, request.language, chat_history)
            
            if decomposition['has_subquestions'] and len(decomposition['subquestions']) <= 3:
                results = await rag_system.process_subquestions_parallel(decomposition['subquestions'])
                all_contexts = []
                for ctx, _, _ in results:
                    if ctx:
                        all_contexts.append(ctx)
                context = "\n\n---\n\n".join(all_contexts)
            else:
                context, _, _ = rag_system.search_context(request.query)
            
            if not context:
                print("hi")
                no_context_msg = rag_system.config['languages'][request.language]['no_context_response']
                yield no_context_msg
                return
            
            # Стриминг ответа
            await rag_system.rate_limiter.wait_for_llm(request.user_id)
            
            full_response = ""
            async for chunk in rag_system.generate_answer_streaming(
                request.query, 
                context, 
                request.language, 
                chat_history
            ):
                full_response += chunk
                yield chunk
            
            # Сохранение после стриминга
            user_msg = schemas.MessageCreate(
                chat_id=request.chat_id,
                user_id=request.user_id,
                message=request.query,
                author="human",
                token_count=rag_system.count_tokens(request.query)
            )
            crud.create_message(db, user_msg)
            
            bot_msg = schemas.MessageCreate(
                chat_id=request.chat_id,
                user_id=request.user_id,
                message=full_response,
                author="bot",
                token_count=rag_system.count_tokens(full_response)
            )
            crud.create_message(db, bot_msg)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\nERROR: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/query/voice")
async def process_voice_query(
    audio: UploadFile = File(...),
    chat_id: str = Form(...),
    user_id: int = Form(...),
    language: str = Form("az"),
    db: Session = Depends(get_db)
):
    """Обработка голосового запроса"""
    temp_audio_path = None
    rag_system = get_rag_system()
    
    try:
        # Проверка типа файла
        allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm']
        if audio.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый формат аудио. Разрешены: {', '.join(allowed_types)}"
            )
        
        # Проверка размера файла (максимум 10MB)
        content = await audio.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Файл слишком большой (макс. 10MB)")
        
        # Сохранение во временный файл
        suffix = os.path.splitext(audio.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        logger.info(f"Saved audio to: {temp_audio_path}, size: {len(content)} bytes")
        
        # Транскрибация
        transcription = await rag_system.transcribe_audio(temp_audio_path)
        
        if not transcription or len(transcription.strip()) == 0:
            raise HTTPException(status_code=400, detail="Не удалось распознать речь")
        
        logger.info(f"Transcription successful: {transcription[:100]}...")
        
        # Обработка транскрибированного текста
        request = schemas.QueryRequest(
            query=transcription,
            chat_id=chat_id,
            user_id=user_id,
            language=language
        )
        
        response = await process_query(request, db)
        
        # Добавляем транскрибированный текст в ответ
        response_dict = response.dict()
        response_dict['transcription'] = transcription
        
        return response_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки голоса: {str(e)}")
    finally:
        # Удаление временного файла
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Removed temp file: {temp_audio_path}")
            except Exception as e:
                logger.error(f"Error removing temp file: {e}")

@app.get("/cache/health")
async def cache_health():
    """Проверка состояния Redis"""
    rag = get_rag_system()
    
    try:
        # Проверка подключения
        rag.cache_manager.redis_client.ping()
        stats = rag.cache_manager.get_cache_stats()
        
        return {
            "status": "healthy",
            "backend": "redis",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# ============= FEEDBACK ENDPOINTS =============
@app.post("/feedback", response_model=schemas.FeedbackResponse)
async def submit_feedback(feedback: schemas.FeedbackCreate, db: Session = Depends(get_db)):
    """Отправка обратной связи"""
    return crud.create_feedback(db, feedback)

@app.get("/users/{user_id}/feedback", response_model=List[schemas.FeedbackResponse])
async def get_user_feedback(user_id: int, db: Session = Depends(get_db)):
    """Получение обратной связи пользователя"""
    return crud.get_user_feedbacks(db, user_id)

# ============= ANALYTICS ENDPOINTS =============
@app.get("/analytics/users/count")
async def get_users_count(db: Session = Depends(get_db)):
    """Количество пользователей"""
    return {"count": crud.get_total_users_count(db)}

@app.get("/analytics/tokens/total")
async def get_total_tokens(db: Session = Depends(get_db)):
    """Общее количество токенов"""
    return {"total": crud.get_total_tokens(db)}

@app.get("/analytics/tokens/average")
async def get_average_tokens(db: Session = Depends(get_db)):
    """Среднее количество токенов"""
    return {"average": crud.get_average_tokens(db)}

@app.post("/analytics/chats/per-day")
async def get_chats_per_day(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Чаты по дням"""
    results = crud.get_chats_per_day(db, filters)
    return {str(day): count for day, count in results}

@app.post("/analytics/questions/per-day")
async def get_questions_per_day(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Вопросы по дням"""
    results = crud.get_questions_per_day(db, filters)
    return {str(day): count for day, count in results}

@app.post("/analytics/tokens/per-day")
async def get_tokens_per_day(
    author: str = None,
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Токены по дням"""
    results = crud.get_token_usage_per_day(db, author, filters)
    return {str(day): tokens for day, tokens in results}

@app.post("/analytics/cost/per-day")
async def get_cost_per_day(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Стоимость по дням"""
    results = crud.get_cost_per_day(db, filters=filters)
    return {str(day): {"cost": cost, "count": count} for day, cost, count in results}

@app.post("/analytics/llm-time/per-day")
async def get_llm_times(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Время ответа LLM по дням"""
    results = crud.get_llm_response_times_per_day(db, filters)
    return {
        str(day): {
            "avg_time": round(float(time), 2) if time else 0.0,  # Açıq float konversiyası
            "count": count
        } 
        for day, time, count in results
    }

@app.post("/analytics/retriever-time/per-day")
async def get_retriever_times(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Время ответа Retriever по дням"""
    results = crud.get_retriever_times_per_day(db, filters)
    return {
        str(day): {
            "avg_time": round(float(time), 2) if time else 0.0,  # Açıq float konversiyası
            "count": count
        } 
        for day, time, count in results
    }

@app.post("/analytics/ratings/per-day")
async def get_ratings_per_day(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Рейтинги по дням"""
    results = crud.get_ratings_per_day(db, filters)
    return {str(day): {"avg_rating": rating, "count": count} for day, rating, count in results}

@app.get("/analytics/questions/today-vs-yesterday")
async def get_questions_comparison(db: Session = Depends(get_db)):
    """Сравнение вопросов сегодня vs вчера"""
    return crud.get_questions_today_vs_yesterday(db)

@app.get("/analytics/questions/per-user")
async def get_avg_questions_per_user(db: Session = Depends(get_db)):
    """Среднее количество вопросов на пользователя"""
    return {"average": crud.get_average_questions_per_user(db)}

@app.get("/analytics/questions/per-chat")
async def get_avg_questions_per_chat(db: Session = Depends(get_db)):
    """Среднее количество вопросов на чат"""
    return {"average": crud.get_average_questions_per_chat(db)}

@app.get("/analytics/cache/stats")
async def get_cache_stats():
    """Статистика кеша"""
    rag_system = get_rag_system()
    query_cache_stats = rag_system.cache_manager.get_cache_stats()
    llm_cache_stats = {
        "llm_prompt_cache_size": len(rag_system.llm_cache.prompt_cache)
    }
    return {**query_cache_stats, **llm_cache_stats}

@app.get("/debug/query-logs")
async def debug_query_logs(db: Session = Depends(get_db)):
    """Debug: Son 5 query log-u göstərir"""
    logs = db.query(models.QueryLog).order_by(models.QueryLog.timestamp.desc()).limit(5).all()
    
    return [
        {
            "id": log.id,
            "query": log.query[:50],
            "llm_time": log.llm_time,
            "llm_time_type": type(log.llm_time).__name__,
            "retriever_time": log.retriever_time,
            "retriever_time_type": type(log.retriever_time).__name__,
            "timestamp": log.timestamp
        }
        for log in logs
    ]

@app.post("/analytics/users/count")
async def get_users_count_filtered(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Количество пользователей с фильтрами"""
    return {"count": crud.get_total_users_count(db, filters)}

@app.post("/analytics/tokens/total")
async def get_total_tokens_filtered(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Общее количество токенов с фильтрами"""
    return {"total": crud.get_total_tokens(db, filters)}

@app.post("/analytics/tokens/average")
async def get_average_tokens_filtered(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Среднее количество токенов с фильтрами"""
    return {"average": crud.get_average_tokens(db, filters)}

@app.post("/analytics/messages/count")
async def get_messages_count_filtered(
    filters: schemas.DashboardFilters = None,
    db: Session = Depends(get_db)
):
    """Общее количество сообщений с фильтрами"""
    return {"count": crud.get_total_messages_count(db, filters)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)