from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import hashlib
import logging
logger = logging.getLogger(__name__)

from app import models, schemas

# ============= USER OPERATIONS =============
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    # Хеширование пароля (в production используйте bcrypt)
    hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
    
    db_user = models.User(
        username=user.username,
        password=hashed_password,
        preferred_language=user.preferred_language
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def verify_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return None
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if user.password == hashed_password:
        return user
    return None

def reset_password(db: Session, username: str, new_password: str):
    user = get_user_by_username(db, username)
    if not user:
        return None
    
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    user.password = hashed_password
    db.commit()
    db.refresh(user)
    return user

# ============= CHAT OPERATIONS =============
def create_chat(db: Session, chat: schemas.ChatCreate):
    # Проверяем существует ли чат с таким chat_id
    existing_chat = db.query(models.Chat).filter(models.Chat.chat_id == chat.chat_id).first()
    if existing_chat:
        logger.info(f"Chat {chat.chat_id} already exists, returning existing")
        return existing_chat
    
    db_chat = models.Chat(
        chat_id=chat.chat_id,
        user_id=chat.user_id,
        header=chat.header,
        language=chat.language
    )
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def get_chat(db: Session, chat_id: str):
    return db.query(models.Chat).filter(models.Chat.chat_id == chat_id).first()

def get_user_chats(db: Session, user_id: int):
    return db.query(models.Chat).filter(models.Chat.user_id == user_id).order_by(models.Chat.created_at.desc()).all()

# ============= MESSAGE OPERATIONS =============
def create_message(db: Session, message: schemas.MessageCreate):
    db_message = models.Message(
        chat_id=message.chat_id,
        user_id=message.user_id,
        message=message.message,
        author=message.author,
        token_count=message.token_count
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_chat_messages(db: Session, chat_id: str):
    return db.query(models.Message).filter(models.Message.chat_id == chat_id).order_by(models.Message.timestamp).all()

# ============= QUERY LOG OPERATIONS =============
def create_query_log(db: Session, log_data: Dict[str, Any]):
    db_log = models.QueryLog(**log_data)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

# ============= FEEDBACK OPERATIONS =============
def create_feedback(db: Session, feedback: schemas.FeedbackCreate):
    db_feedback = models.Feedback(
        user_id=feedback.user_id,
        chat_id=feedback.chat_id,
        query_log_id=feedback.query_log_id,
        rating=feedback.rating,
        comment=feedback.comment
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

def get_user_feedbacks(db: Session, user_id: int):
    return db.query(models.Feedback).filter(models.Feedback.user_id == user_id).all()

# ============= ANALYTICS =============
def get_total_users_count(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Общее количество уникальных пользователей"""
    query = db.query(func.count(func.distinct(models.Message.user_id))).filter(
        models.Message.author == 'human'
    )
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Message.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Message.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Message.user_id == user_id)
            else:
                return 0
    
    return query.scalar() or 0

def get_total_tokens(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Общее количество токенов"""
    query = db.query(func.sum(models.Message.token_count))
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Message.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Message.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Message.user_id == user_id)
            else:
                return 0
    
    return query.scalar() or 0

def get_average_tokens(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Средний счет токенов на сообщение"""
    query = db.query(func.avg(models.Message.token_count))
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Message.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Message.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Message.user_id == user_id)
            else:
                return 0
    
    return query.scalar() or 0

def get_user_id_by_username(db: Session, username: str) -> Optional[int]:
    """Получить user_id по username"""
    user = db.query(models.User).filter(models.User.username == username).first()
    return user.id if user else None

def get_chats_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Количество чатов по дням"""
    query = db.query(
        func.date(models.Chat.created_at).label("day"),
        func.count(models.Chat.id).label("count")
    )
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Chat.created_at >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Chat.created_at <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Chat.user_id == user_id)
            else:
                return []
    
    return query.group_by(func.date(models.Chat.created_at)).all()

def get_questions_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Количество вопросов по дням"""
    query = db.query(
        func.date(models.Message.timestamp).label("day"),
        func.count(models.Message.id).label("count")
    ).filter(models.Message.author == "human")
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Message.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Message.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Message.user_id == user_id)
            else:
                return []
    
    return query.group_by(func.date(models.Message.timestamp)).all()

def get_token_usage_per_day(db: Session, author: str = None, filters: Optional[schemas.DashboardFilters] = None):
    """Использование токенов по дням"""
    query = db.query(
        func.date(models.Message.timestamp).label("day"),
        func.sum(models.Message.token_count).label("tokens")
    )
    
    if author:
        query = query.filter(models.Message.author == author)
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Message.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Message.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Message.user_id == user_id)
            else:
                return []
    
    return query.group_by(func.date(models.Message.timestamp)).all()

def get_llm_response_times_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Время ответа LLM по дням"""
    query = db.query(
        func.date(models.QueryLog.timestamp).label("day"),
        func.round(func.avg(models.QueryLog.llm_time), 2).label("avg_time"),
        func.count(models.QueryLog.id).label("count")
    )
    
    if filters:
        if filters.start_date:
            query = query.filter(models.QueryLog.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.QueryLog.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.QueryLog.user_id == user_id)
            else:
                return []
        if filters.min_rating or filters.max_rating:
            query = query.join(models.Feedback, models.QueryLog.id == models.Feedback.query_log_id)
            if filters.min_rating:
                query = query.filter(models.Feedback.rating >= filters.min_rating)
            if filters.max_rating:
                query = query.filter(models.Feedback.rating <= filters.max_rating)
    
    return query.group_by(func.date(models.QueryLog.timestamp)).all()

def get_retriever_times_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Время ответа Retriever по дням"""
    query = db.query(
        func.date(models.QueryLog.timestamp).label("day"),
        func.round(func.avg(models.QueryLog.retriever_time), 2).label("avg_time"),
        func.count(models.QueryLog.id).label("count")
    ).filter(models.QueryLog.retriever_used == True)
    
    if filters:
        if filters.start_date:
            query = query.filter(models.QueryLog.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.QueryLog.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.QueryLog.user_id == user_id)
            else:
                return []
        if filters.min_rating or filters.max_rating:
            query = query.join(models.Feedback, models.QueryLog.id == models.Feedback.query_log_id)
            if filters.min_rating:
                query = query.filter(models.Feedback.rating >= filters.min_rating)
            if filters.max_rating:
                query = query.filter(models.Feedback.rating <= filters.max_rating)
    
    return query.group_by(func.date(models.QueryLog.timestamp)).all()

def get_ratings_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Рейтинги по дням"""
    query = db.query(
        func.date(models.Feedback.timestamp).label("day"),
        func.avg(models.Feedback.rating).label("avg_rating"),
        func.count(models.Feedback.id).label("count")
    )
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Feedback.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Feedback.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Feedback.user_id == user_id)
            else:
                return []
    
    return query.group_by(func.date(models.Feedback.timestamp)).all()

def get_cost_per_day(db: Session, cost_per_million: float = 15.0, filters: Optional[schemas.DashboardFilters] = None):
    """Стоимость по дням"""
    query = db.query(
        func.date(models.QueryLog.timestamp).label("day"),
        func.sum(models.QueryLog.llm_cost).label("total_cost"),
        func.count(models.QueryLog.id).label("count")
    )
    
    if filters:
        if filters.start_date:
            query = query.filter(models.QueryLog.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.QueryLog.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.QueryLog.user_id == user_id)
            else:
                return []
    
    return query.group_by(func.date(models.QueryLog.timestamp)).all()

def get_questions_today_vs_yesterday(db: Session):
    """Сравнение вопросов сегодня vs вчера"""
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    today_count = db.query(func.count(models.Message.id)).filter(
        models.Message.author == 'human',
        func.date(models.Message.timestamp) == today
    ).scalar() or 0
    
    yesterday_count = db.query(func.count(models.Message.id)).filter(
        models.Message.author == 'human',
        func.date(models.Message.timestamp) == yesterday
    ).scalar() or 0
    
    return {"today": today_count, "yesterday": yesterday_count}

def get_average_questions_per_user(db: Session):
    """Среднее количество вопросов на пользователя"""
    subquery = db.query(
        models.Message.user_id,
        func.count(models.Message.id).label('count')
    ).filter(models.Message.author == 'human').group_by(models.Message.user_id).subquery()
    
    return db.query(func.avg(subquery.c.count)).scalar() or 0

def get_average_questions_per_chat(db: Session):
    """Среднее количество вопросов на чат"""
    subquery = db.query(
        models.Message.chat_id,
        func.count(models.Message.id).label('count')
    ).filter(models.Message.author == 'human').group_by(models.Message.chat_id).subquery()
    
    return db.query(func.avg(subquery.c.count)).scalar() or 0

def get_total_messages_count(db: Session, filters: Optional[schemas.DashboardFilters] = None):
    """Общее количество сообщений"""
    query = db.query(func.count(models.Message.id))
    
    if filters:
        if filters.start_date:
            query = query.filter(models.Message.timestamp >= filters.start_date)
        if filters.end_date:
            query = query.filter(models.Message.timestamp <= filters.end_date)
        if filters.username:
            user_id = get_user_id_by_username(db, filters.username)
            if user_id:
                query = query.filter(models.Message.user_id == user_id)
            else:
                return 0
    
    return query.scalar() or 0