from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Boolean, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    """Модель пользователя"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    preferred_language = Column(String, default="az")  # az, ru, en
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")

class Chat(Base):
    """Модель чата"""
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, unique=True, index=True, nullable=False)
    header = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    language = Column(String, default="az")
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    """Модель сообщения"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.chat_id"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    author = Column(String, index=True)  # 'human' or 'bot'
    token_count = Column(Integer, default=0)
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="messages")
    chat = relationship("Chat", back_populates="messages")

class QueryLog(Base):
    """Расширенное логирование запросов"""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Query details
    query = Column(Text, nullable=False)
    language = Column(String, index=True)
    
    # Retriever details
    retriever_time = Column(Float)  # секунды
    top_k_chunks = Column(JSON)  # [{"text": "...", "similarity": 0.85}, ...]
    avg_similarity = Column(Float)
    retriever_used = Column(Boolean, default=True)
    
    # LLM details
    llm_response = Column(Text)
    llm_time = Column(Float)  # секунды
    llm_tokens = Column(Integer)
    llm_cost = Column(Float)  # USD
    llm_confidence = Column(Float)  # 0-1
    
    # Chain of Thought
    had_subquestions = Column(Boolean, default=False)
    subquestions = Column(JSON)  # ["subq1", "subq2"]
    requires_context = Column(Boolean, default=False)  
    context_reference = Column(Text)               
    cot_total_time = Column(Float)
    cot_total_cost = Column(Float)
    
    # Status
    was_cached = Column(Boolean, default=False)
    had_retry = Column(Boolean, default=False)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    is_fraud = Column(Boolean, default=False)
    fraud_reason = Column(Text)
    
    # Production issues
    debug_info = Column(JSON)
    
    timestamp = Column(DateTime, default=func.now(), index=True)

class Feedback(Base):
    """Оценки пользователей"""
    __tablename__ = "feedbacks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chat_id = Column(String, index=True)
    query_log_id = Column(Integer, ForeignKey("query_logs.id"))
    
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text)
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="feedbacks")