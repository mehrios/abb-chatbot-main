"""
Pydantic schemas for API request/response validation.

This module defines all data models used for API endpoints including:
- User management (registration, authentication, password reset)
- Chat management (creation, retrieval)
- Message handling (storage, retrieval)
- Query processing (text and voice)
- Feedback collection
- Analytics and dashboard filtering
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# USER SCHEMAS
# ============================================================================

class UserCreate(BaseModel):
    """
    Schema for creating a new user account.
    
    Attributes:
        username: Unique username (3-50 characters)
        password: User password (minimum 6 characters)
        preferred_language: User's preferred language (az/ru/en)
    """
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    preferred_language: str = Field(default="az", pattern="^(az|ru|en)$")


class UserLogin(BaseModel):
    """
    Schema for user authentication.
    
    Attributes:
        username: User's username
        password: User's password
    """
    username: str
    password: str


class UserResponse(BaseModel):
    """
    Schema for user data in API responses.
    
    Attributes:
        id: Unique user identifier
        username: User's username
        preferred_language: User's preferred language setting
        created_at: Account creation timestamp
    """
    id: int
    username: str
    preferred_language: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class PasswordReset(BaseModel):
    """
    Schema for password reset operation.
    
    Attributes:
        username: Username for password reset
        new_password: New password (minimum 6 characters)
    """
    username: str
    new_password: str = Field(..., min_length=6)


# ============================================================================
# CHAT SCHEMAS
# ============================================================================

class ChatCreate(BaseModel):
    """
    Schema for creating a new chat session.
    
    Attributes:
        chat_id: Unique chat identifier
        user_id: ID of the user creating the chat
        header: Chat title or header
        language: Chat language setting (az/ru/en)
    """
    chat_id: str
    user_id: int
    header: str
    language: str = Field(default="az", pattern="^(az|ru|en)$")


class ChatResponse(BaseModel):
    """
    Schema for chat data in API responses.
    
    Attributes:
        id: Unique database identifier
        chat_id: Chat session identifier
        header: Chat title
        language: Chat language setting
        created_at: Chat creation timestamp
    """
    id: int
    chat_id: str
    header: str
    language: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# MESSAGE SCHEMAS
# ============================================================================

class MessageCreate(BaseModel):
    """
    Schema for creating a new message in a chat.
    
    Attributes:
        chat_id: Associated chat identifier
        user_id: ID of the user sending the message
        message: Message content
        author: Message author type ('human' or 'bot')
        token_count: Number of tokens in the message
    """
    chat_id: str
    user_id: int
    message: str
    author: str  # 'human' or 'bot'
    token_count: int = 0


class MessageResponse(BaseModel):
    """
    Schema for message data in API responses.
    
    Attributes:
        id: Unique message identifier
        chat_id: Associated chat identifier
        message: Message content
        author: Message author type
        token_count: Token count for the message
        timestamp: Message creation timestamp
    """
    id: int
    chat_id: str
    message: str
    author: str
    token_count: int
    timestamp: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# QUERY SCHEMAS
# ============================================================================

class QueryRequest(BaseModel):
    """
    Schema for processing a text query request.
    
    Attributes:
        query: User's query text (minimum 1 character)
        chat_id: Associated chat identifier
        user_id: ID of the requesting user
        language: Query language (az/ru/en)
    """
    query: str = Field(..., min_length=1)
    chat_id: str
    user_id: int
    language: str = Field(default="az", pattern="^(az|ru|en)$")


class QueryResponse(BaseModel):
    """
    Schema for query processing response.
    
    Attributes:
        success: Whether the query was processed successfully
        response: Generated response text
        error: Error message if processing failed
        processing_time: Time taken to process the query
        had_subquestions: Whether the query was decomposed into subquestions
        subquestions: List of generated subquestions
        requires_context: Whether additional context is needed
        context_reference: Reference to required context
        was_cached: Whether the response was served from cache
        query_log_id: Database identifier for the query log
    """
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    had_subquestions: bool = False
    subquestions: Optional[List[str]] = []
    requires_context: bool = False
    context_reference: Optional[str] = None
    was_cached: bool = False
    query_log_id: Optional[int] = None


class VoiceQueryRequest(BaseModel):
    """
    Schema for processing a voice query request.
    
    Attributes:
        audio_data: Base64 encoded audio data
        chat_id: Associated chat identifier
        user_id: ID of the requesting user
        language: Query language (az/ru/en)
    """
    audio_data: str  # base64 encoded
    chat_id: str
    user_id: int
    language: str = Field(default="az", pattern="^(az|ru|en)$")


class VoiceQueryResponse(QueryResponse):
    """
    Schema for voice query response with transcription.
    
    Extends QueryResponse with additional voice-specific attributes.
    
    Attributes:
        transcription: Transcribed text from the audio input
    """
    transcription: Optional[str] = None


# ============================================================================
# FEEDBACK SCHEMAS
# ============================================================================

class FeedbackCreate(BaseModel):
    """
    Schema for creating user feedback.
    
    Attributes:
        user_id: ID of the user providing feedback
        chat_id: Associated chat identifier
        query_log_id: Optional query log reference
        rating: Feedback rating (1-5 scale)
        comment: Optional feedback comment
    """
    user_id: int
    chat_id: str
    query_log_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """
    Schema for feedback data in API responses.
    
    Attributes:
        id: Unique feedback identifier
        rating: Feedback rating value
        comment: Optional feedback comment
        timestamp: Feedback submission timestamp
    """
    id: int
    rating: int
    comment: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# ANALYTICS SCHEMAS
# ============================================================================

class AnalyticsResponse(BaseModel):
    """
    Schema for analytics data responses.
    
    Attributes:
        data: Dictionary containing analytics metrics and values
    """
    data: Dict[str, Any]


class DashboardFilters(BaseModel):
    """
    Schema for dashboard data filtering options.
    
    Attributes:
        start_date: Filter start date (inclusive)
        end_date: Filter end date (inclusive)
        username: Filter by specific username
        min_rating: Minimum rating filter (1-5)
        max_rating: Maximum rating filter (1-5)
    """
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    username: Optional[str] = None
    min_rating: Optional[int] = Field(None, ge=1, le=5)
    max_rating: Optional[int] = Field(None, ge=1, le=5)