"""
AutoPulse Configuration
Loads settings from environment variables
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database
    database_url: str = "postgresql+asyncpg://autopulse:autopulse_secret@127.0.0.1:5433/autopulse"
    
    # For sync operations (alembic)
    database_url_sync: str = "postgresql://autopulse:autopulse_secret@127.0.0.1:5433/autopulse"
    
    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    debug: bool = True
    
    # Simulator
    simulator_interval_ms: int = 1000
    vehicle_vin: str = "WP0AB2A91NS123456"
    
    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()
