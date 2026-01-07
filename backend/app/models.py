from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Float,
    Boolean,
    Text,
    JSON
)
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.sql import func

from app.database import Base


# ============================================================================
# USER MODEL
# ============================================================================

class User(Base):
    """
    User model for authentication and profile management.
    
    Attributes:
        id: Unique user identifier
        username: Unique username for authentication
        password: Hashed password
        preferred_language: User's preferred language (az, ru, en)
        created_at: Account creation timestamp
    """
    __tablename__ = "users"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    username: Mapped[str] = Column(String, unique=True, index=True, nullable=False)
    password: Mapped[str] = Column(String, nullable=False)
    preferred_language: Mapped[str] = Column(String, default="az")  # az, ru, en
    created_at: Mapped[datetime] = Column(DateTime, default=func.now())
    
    # Relationships
    chats: Mapped[List["Chat"]] = relationship(
        "Chat",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship(
        "Feedback",
        back_populates="user",
        cascade="all, delete-orphan"
    )


# ============================================================================
# CHAT MODEL
# ============================================================================

class Chat(Base):
    """
    Chat session model for organizing conversations.
    
    Attributes:
        id: Unique chat identifier (internal)
        chat_id: External chat identifier (UUID)
        header: Chat title/header
        user_id: Foreign key to user
        language: Chat language preference
        created_at: Chat creation timestamp
    """
    __tablename__ = "chats"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    chat_id: Mapped[str] = Column(String, unique=True, index=True, nullable=False)
    header: Mapped[Optional[str]] = Column(String, index=True)
    user_id: Mapped[int] = Column(Integer, ForeignKey("users.id"), nullable=False)
    language: Mapped[str] = Column(String, default="az")
    created_at: Mapped[datetime] = Column(DateTime, default=func.now())
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chats")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan"
    )


# ============================================================================
# MESSAGE MODEL
# ============================================================================

class Message(Base):
    """
    Message model for storing chat conversations.
    
    Attributes:
        id: Unique message identifier
        chat_id: Foreign key to chat
        user_id: Foreign key to user
        message: Message content
        author: Message author ('human' or 'bot')
        token_count: Number of tokens in message
        timestamp: Message timestamp
    """
    __tablename__ = "messages"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    chat_id: Mapped[str] = Column(String, ForeignKey("chats.chat_id"), index=True, nullable=False)
    user_id: Mapped[int] = Column(Integer, ForeignKey("users.id"), nullable=False)
    message: Mapped[str] = Column(Text, nullable=False)
    author: Mapped[str] = Column(String, index=True)  # 'human' or 'bot'
    token_count: Mapped[int] = Column(Integer, default=0)
    timestamp: Mapped[datetime] = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="messages")
    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")


# ============================================================================
# QUERY LOG MODEL
# ============================================================================

class QueryLog(Base):
    """
    Extended query logging model for analytics and monitoring.
    
    Attributes:
        id: Unique log identifier
        chat_id: Associated chat identifier
        user_id: Foreign key to user
        
        Query Details:
            query: User query text
            language: Query language
        
        Retriever Details:
            retriever_time: Time spent on retrieval (seconds)
            top_k_chunks: Retrieved chunks with similarity scores
            avg_similarity: Average similarity score
            retriever_used: Whether retriever was used
        
        LLM Details:
            llm_response: Generated response
            llm_time: Time spent on LLM generation (seconds)
            llm_tokens: Number of tokens used
            llm_cost: Cost in USD
            llm_confidence: Response confidence score (0-1)
        
        Chain of Thought:
            had_subquestions: Whether query was decomposed
            subquestions: List of subquestions
            requires_context: Whether context was needed
            context_reference: Context reference information
            cot_total_time: Total CoT processing time
            cot_total_cost: Total CoT cost
        
        Status:
            was_cached: Whether response was cached
            had_retry: Whether retry was needed
            retry_count: Number of retries
            error_message: Error details if failed
            is_fraud: Fraud detection flag
            fraud_reason: Fraud detection reason
        
        Debug:
            debug_info: Additional debug information
            timestamp: Log timestamp
    """
    __tablename__ = "query_logs"
    
    # Primary identification
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    chat_id: Mapped[Optional[str]] = Column(String, index=True)
    user_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("users.id"))
    
    # Query details
    query: Mapped[str] = Column(Text, nullable=False)
    language: Mapped[Optional[str]] = Column(String, index=True)
    
    # Retriever details
    retriever_time: Mapped[Optional[float]] = Column(Float)  # seconds
    top_k_chunks: Mapped[Optional[dict]] = Column(JSON)  # [{"text": "...", "similarity": 0.85}, ...]
    avg_similarity: Mapped[Optional[float]] = Column(Float)
    retriever_used: Mapped[bool] = Column(Boolean, default=True)
    
    # LLM details
    llm_response: Mapped[Optional[str]] = Column(Text)
    llm_time: Mapped[Optional[float]] = Column(Float)  # seconds
    llm_tokens: Mapped[Optional[int]] = Column(Integer)
    llm_cost: Mapped[Optional[float]] = Column(Float)  # USD
    llm_confidence: Mapped[Optional[float]] = Column(Float)  # 0-1
    
    # Chain of Thought
    had_subquestions: Mapped[bool] = Column(Boolean, default=False)
    subquestions: Mapped[Optional[list]] = Column(JSON)  # ["subq1", "subq2"]
    requires_context: Mapped[bool] = Column(Boolean, default=False)
    context_reference: Mapped[Optional[str]] = Column(Text)
    cot_total_time: Mapped[Optional[float]] = Column(Float)
    cot_total_cost: Mapped[Optional[float]] = Column(Float)
    
    # Status flags
    was_cached: Mapped[bool] = Column(Boolean, default=False)
    had_retry: Mapped[bool] = Column(Boolean, default=False)
    retry_count: Mapped[int] = Column(Integer, default=0)
    error_message: Mapped[Optional[str]] = Column(Text)
    is_fraud: Mapped[bool] = Column(Boolean, default=False)
    fraud_reason: Mapped[Optional[str]] = Column(Text)
    
    # Debug information
    debug_info: Mapped[Optional[dict]] = Column(JSON)
    timestamp: Mapped[datetime] = Column(DateTime, default=func.now(), index=True)


# ============================================================================
# FEEDBACK MODEL
# ============================================================================

class Feedback(Base):
    """
    User feedback model for rating and comments.
    
    Attributes:
        id: Unique feedback identifier
        user_id: Foreign key to user
        chat_id: Associated chat identifier
        query_log_id: Associated query log identifier
        rating: User rating (1-5)
        comment: Optional feedback comment
        timestamp: Feedback timestamp
    """
    __tablename__ = "feedbacks"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = Column(Integer, ForeignKey("users.id"), nullable=False)
    chat_id: Mapped[Optional[str]] = Column(String, index=True)
    query_log_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("query_logs.id"))
    
    rating: Mapped[int] = Column(Integer, nullable=False)  # 1-5
    comment: Mapped[Optional[str]] = Column(Text)
    timestamp: Mapped[datetime] = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="feedbacks")