"""
AutoPulse Database Configuration
Async SQLAlchemy setup for PostgreSQL + TimescaleDB
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

# Async engine for PostgreSQL
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    connect_args={
        "ssl": False,
    },
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# Base class for all models
class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """
    Dependency that provides a database session.
    Use with FastAPI's Depends().
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database connection (called on startup)"""
    async with engine.begin() as conn:
        pass