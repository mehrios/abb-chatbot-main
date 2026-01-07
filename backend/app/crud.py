"""
Database CRUD operations for user management, chat handling, and analytics.

This module provides comprehensive database operations including:
- User authentication and management
- Chat and message handling
- Query logging and feedback collection
- Analytics and reporting functions
"""

import hashlib
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from app import models, schemas

logger = logging.getLogger(__name__)


# ============================================================================
# USER OPERATIONS
# ============================================================================

def get_user(db: Session, user_id: int) -> Optional[models.User]:
    """
    Retrieve user by ID.
    
    Args:
        db: Database session
        user_id: User's unique identifier
        
    Returns:
        User object if found, None otherwise
    """
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> Optional[models.User]:
    """
    Retrieve user by username.
    
    Args:
        db: Database session
        username: User's username
        
    Returns:
        User object if found, None otherwise
    """
    return db.query(models.User).filter(models.User.username == username).first()


def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    """
    Create new user with hashed password.
    
    Note: In production, use bcrypt instead of SHA256 for password hashing.
    
    Args:
        db: Database session
        user: User creation schema with username and password
        
    Returns:
        Created user object
    """
    # Hash password (use bcrypt in production)
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


def verify_user(db: Session, username: str, password: str) -> Optional[models.User]:
    """
    Verify user credentials for authentication.
    
    Args:
        db: Database session
        username: User's username
        password: Plain text password to verify
        
    Returns:
        User object if credentials are valid, None otherwise
    """
    user = get_user_by_username(db, username)
    if not user:
        return None
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if user.password == hashed_password:
        return user
    return None


def reset_password(db: Session, username: str, new_password: str) -> Optional[models.User]:
    """
    Reset user's password.
    
    Args:
        db: Database session
        username: User's username
        new_password: New plain text password
        
    Returns:
        Updated user object if user exists, None otherwise
    """
    user = get_user_by_username(db, username)
    if not user:
        return None
    
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    user.password = hashed_password
    db.commit()
    db.refresh(user)
    return user


# ============================================================================
# CHAT OPERATIONS
# ============================================================================

def create_chat(db: Session, chat: schemas.ChatCreate) -> models.Chat:
    """
    Create new chat or return existing one.
    
    Args:
        db: Database session
        chat: Chat creation schema
        
    Returns:
        Created or existing chat object
    """
    # Check if chat with this chat_id already exists
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


def get_chat(db: Session, chat_id: str) -> Optional[models.Chat]:
    """
    Retrieve chat by chat_id.
    
    Args:
        db: Database session
        chat_id: Chat's unique identifier
        
    Returns:
        Chat object if found, None otherwise
    """
    return db.query(models.Chat).filter(models.Chat.chat_id == chat_id).first()


def get_user_chats(db: Session, user_id: int) -> List[models.Chat]:
    """
    Retrieve all chats for a specific user, ordered by creation date.
    
    Args:
        db: Database session
        user_id: User's unique identifier
        
    Returns:
        List of chat objects ordered by creation date (newest first)
    """
    return db.query(models.Chat).filter(
        models.Chat.user_id == user_id
    ).order_by(models.Chat.created_at.desc()).all()


# ============================================================================
# MESSAGE OPERATIONS
# ============================================================================

def create_message(db: Session, message: schemas.MessageCreate) -> models.Message:
    """
    Create new message in a chat.
    
    Args:
        db: Database session
        message: Message creation schema
        
    Returns:
        Created message object
    """
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


def get_chat_messages(db: Session, chat_id: str) -> List[models.Message]:
    """
    Retrieve all messages in a chat, ordered by timestamp.
    
    Args:
        db: Database session
        chat_id: Chat's unique identifier
        
    Returns:
        List of message objects ordered by timestamp
    """
    return db.query(models.Message).filter(
        models.Message.chat_id == chat_id
    ).order_by(models.Message.timestamp).all()


# ============================================================================
# QUERY LOG OPERATIONS
# ============================================================================

def create_query_log(db: Session, log_data: Dict[str, Any]) -> models.QueryLog:
    """
    Create query log entry for analytics.
    
    Args:
        db: Database session
        log_data: Dictionary containing log data fields
        
    Returns:
        Created query log object
    """
    db_log = models.QueryLog(**log_data)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log


# ============================================================================
# FEEDBACK OPERATIONS
# ============================================================================

def create_feedback(db: Session, feedback: schemas.FeedbackCreate) -> models.Feedback:
    """
    Create user feedback for a query.
    
    Args:
        db: Database session
        feedback: Feedback creation schema
        
    Returns:
        Created feedback object
    """
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


def get_user_feedbacks(db: Session, user_id: int) -> List[models.Feedback]:
    """
    Retrieve all feedback submitted by a user.
    
    Args:
        db: Database session
        user_id: User's unique identifier
        
    Returns:
        List of feedback objects
    """
    return db.query(models.Feedback).filter(models.Feedback.user_id == user_id).all()


# ============================================================================
# ANALYTICS - BASIC METRICS
# ============================================================================

def get_user_id_by_username(db: Session, username: str) -> Optional[int]:
    """
    Get user ID by username.
    
    Args:
        db: Database session
        username: User's username
        
    Returns:
        User ID if found, None otherwise
    """
    user = db.query(models.User).filter(models.User.username == username).first()
    return user.id if user else None


def get_total_users_count(db: Session, filters: Optional[schemas.DashboardFilters] = None) -> int:
    """
    Get total count of unique users who sent messages.
    
    Args:
        db: Database session
        filters: Optional filters for date range and username
        
    Returns:
        Count of unique users
    """
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


def get_total_tokens(db: Session, filters: Optional[schemas.DashboardFilters] = None) -> int:
    """
    Get total token count across all messages.
    
    Args:
        db: Database session
        filters: Optional filters for date range and username
        
    Returns:
        Total token count
    """
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


def get_average_tokens(db: Session, filters: Optional[schemas.DashboardFilters] = None) -> float:
    """
    Get average token count per message.
    
    Args:
        db: Database session
        filters: Optional filters for date range and username
        
    Returns:
        Average token count per message
    """
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


def get_total_messages_count(db: Session, filters: Optional[schemas.DashboardFilters] = None) -> int:
    """
    Get total count of all messages.
    
    Args:
        db: Database session
        filters: Optional filters for date range and username
        
    Returns:
        Total message count
    """
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


# ============================================================================
# ANALYTICS - TIME-BASED METRICS
# ============================================================================

def get_chats_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None) -> List[tuple]:
    """
    Get count of chats created per day.
    
    Args:
        db: Database session
        filters: Optional filters for date range and username
        
    Returns:
        List of tuples (day, count) grouped by date
    """
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


def get_questions_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None) -> List[tuple]:
    """
    Get count of user questions per day.
    
    Args:
        db: Database session
        filters: Optional filters for date range and username
        
    Returns:
        List of tuples (day, count) grouped by date
    """
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


def get_token_usage_per_day(
    db: Session, 
    author: Optional[str] = None, 
    filters: Optional[schemas.DashboardFilters] = None
) -> List[tuple]:
    """
    Get token usage per day, optionally filtered by author.
    
    Args:
        db: Database session
        author: Optional filter for message author ('human' or 'bot')
        filters: Optional filters for date range and username
        
    Returns:
        List of tuples (day, total_tokens) grouped by date
    """
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


def get_llm_response_times_per_day(
    db: Session, 
    filters: Optional[schemas.DashboardFilters] = None
) -> List[tuple]:
    """
    Get average LLM response time per day.
    
    Args:
        db: Database session
        filters: Optional filters for date range, username, and rating range
        
    Returns:
        List of tuples (day, avg_time, count) grouped by date
    """
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


def get_retriever_times_per_day(
    db: Session, 
    filters: Optional[schemas.DashboardFilters] = None
) -> List[tuple]:
    """
    Get average retriever response time per day (only when retriever was used).
    
    Args:
        db: Database session
        filters: Optional filters for date range, username, and rating range
        
    Returns:
        List of tuples (day, avg_time, count) grouped by date
    """
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


def get_ratings_per_day(db: Session, filters: Optional[schemas.DashboardFilters] = None) -> List[tuple]:
    """
    Get average rating per day.
    
    Args:
        db: Database session
        filters: Optional filters for date range and username
        
    Returns:
        List of tuples (day, avg_rating, count) grouped by date
    """
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


def get_cost_per_day(
    db: Session, 
    cost_per_million: float = 15.0, 
    filters: Optional[schemas.DashboardFilters] = None
) -> List[tuple]:
    """
    Get total cost per day based on LLM usage.
    
    Args:
        db: Database session
        cost_per_million: Cost per million tokens (default: 15.0)
        filters: Optional filters for date range and username
        
    Returns:
        List of tuples (day, total_cost, count) grouped by date
    """
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


# ============================================================================
# ANALYTICS - COMPARATIVE METRICS
# ============================================================================

def get_questions_today_vs_yesterday(db: Session) -> Dict[str, int]:
    """
    Compare question counts between today and yesterday.
    
    Args:
        db: Database session
        
    Returns:
        Dictionary with 'today' and 'yesterday' counts
    """
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


def get_average_questions_per_user(db: Session) -> float:
    """
    Calculate average number of questions per user.
    
    Args:
        db: Database session
        
    Returns:
        Average questions per user
    """
    subquery = db.query(
        models.Message.user_id,
        func.count(models.Message.id).label('count')
    ).filter(models.Message.author == 'human').group_by(models.Message.user_id).subquery()
    
    return db.query(func.avg(subquery.c.count)).scalar() or 0


def get_average_questions_per_chat(db: Session) -> float:
    """
    Calculate average number of questions per chat.
    
    Args:
        db: Database session
        
    Returns:
        Average questions per chat
    """
    subquery = db.query(
        models.Message.chat_id,
        func.count(models.Message.id).label('count')
    ).filter(models.Message.author == 'human').group_by(models.Message.chat_id).subquery()
    
    return db.query(func.avg(subquery.c.count)).scalar() or 0