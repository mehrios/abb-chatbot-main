from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./abb.db")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency для получения сессии БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def migrate_database():
    """Автоматическая миграция для новых колонок"""
    try:
        inspector = inspect(engine)
        
        # Проверяем существование таблицы query_logs
        if 'query_logs' not in inspector.get_table_names():
            logger.info("Table query_logs doesn't exist yet, skipping migration")
            return
        
        columns = [col['name'] for col in inspector.get_columns('query_logs')]
        
        with engine.begin() as conn:
            # Добавляем requires_context если её нет
            if 'requires_context' not in columns:
                conn.execute(text("""
                    ALTER TABLE query_logs 
                    ADD COLUMN requires_context BOOLEAN DEFAULT 0
                """))
                logger.info("✓ Added requires_context column")
                print("✓ Added requires_context column")
            
            # Добавляем context_reference если её нет
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

def init_db():
    """Инициализация базы данных"""
    from app.models import User, Chat, Message, QueryLog, Feedback
    
    # Создаем все таблицы
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")
    
    # Запускаем миграцию для новых колонок
    migrate_database()
    
    print("✓ Database initialized")