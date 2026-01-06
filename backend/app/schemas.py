from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ============= USER SCHEMAS =============
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    preferred_language: str = Field(default="az", pattern="^(az|ru|en)$")

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    preferred_language: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class PasswordReset(BaseModel):
    username: str
    new_password: str = Field(..., min_length=6)

# ============= CHAT SCHEMAS =============
class ChatCreate(BaseModel):
    chat_id: str
    user_id: int
    header: str
    language: str = Field(default="az", pattern="^(az|ru|en)$")

class ChatResponse(BaseModel):
    id: int
    chat_id: str
    header: str
    language: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# ============= MESSAGE SCHEMAS =============
class MessageCreate(BaseModel):
    chat_id: str
    user_id: int
    message: str
    author: str  # 'human' or 'bot'
    token_count: int = 0

class MessageResponse(BaseModel):
    id: int
    chat_id: str
    message: str
    author: str
    token_count: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

# ============= QUERY SCHEMAS =============
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    chat_id: str
    user_id: int
    language: str = Field(default="az", pattern="^(az|ru|en)$")

class QueryResponse(BaseModel):
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
    audio_data: str  # base64 encoded
    chat_id: str
    user_id: int
    language: str = Field(default="az", pattern="^(az|ru|en)$")

class VoiceQueryResponse(QueryResponse):
    """Ответ на голосовой запрос с транскрипцией"""
    transcription: Optional[str] = None

# ============= FEEDBACK SCHEMAS =============
class FeedbackCreate(BaseModel):
    user_id: int
    chat_id: str
    query_log_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    rating: int
    comment: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True

# ============= ANALYTICS SCHEMAS =============
class AnalyticsResponse(BaseModel):
    data: Dict[str, Any]

class DashboardFilters(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    username: Optional[str] = None 
    min_rating: Optional[int] = Field(None, ge=1, le=5)
    max_rating: Optional[int] = Field(None, ge=1, le=5)