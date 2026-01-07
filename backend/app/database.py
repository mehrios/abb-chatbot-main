"""
Database configuration and initialization module.

This module handles:
- Database connection setup and configuration
- Session management
- Database migrations
- Schema initialization
"""

import os
import logging
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./abb.db")

# Create database engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.
    
    Provides a database session that automatically closes after use.
    Use with FastAPI's Depends() for automatic session management.
    
    Yields:
        Database session
        
    Example:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# DATABASE MIGRATION
# ============================================================================

def migrate_database() -> None:
    """
    Perform automatic database migration for new columns.
    
    Adds missing columns to existing tables without dropping data.
    Currently handles:
    - query_logs.requires_context (BOOLEAN)
    - query_logs.context_reference (TEXT)
    
    Note: This is a simple migration system. For production, consider
    using Alembic for more robust schema migrations.
    """
    try:
        inspector = inspect(engine)
        
        # Check if query_logs table exists
        if 'query_logs' not in inspector.get_table_names():
            logger.info("Table query_logs doesn't exist yet, skipping migration")
            return
        
        # Get existing column names
        columns = [col['name'] for col in inspector.get_columns('query_logs')]
        
        with engine.begin() as conn:
            # Add requires_context column if missing
            if 'requires_context' not in columns:
                conn.execute(text("""
                    ALTER TABLE query_logs 
                    ADD COLUMN requires_context BOOLEAN DEFAULT 0
                """))
                logger.info("✓ Added requires_context column")
                print("✓ Added requires_context column")
            
            # Add context_reference column if missing
            if 'context_reference' not in columns:
                conn.execute(text("""
                    ALTER TABLE query_logs 
                    ADD COLUMN context_reference TEXT
                """))
                logger.info("✓ Added context_reference column")
                print("✓ Added context_reference column")
        
        logger.info("✓ Database migration completed")
        
    except Exception as e:
        logger.error(f"Migration error: {e}")
        print(f"✗ Migration error: {e}")


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db() -> None:
    """
    Initialize database with all tables and run migrations.
    
    This function:
    1. Creates all tables defined in models
    2. Runs migrations to add any missing columns
    3. Ensures database schema is up to date
    
    Should be called on application startup.
    """
    # Import models to register them with Base
    from app.models import User, Chat, Message, QueryLog, Feedback
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")
    
    # Run migrations for new columns
    migrate_database()
    
    print("✓ Database initialized")