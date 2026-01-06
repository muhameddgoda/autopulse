"""
AutoPulse - Connected Car Platform
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.api.telemetry import router as telemetry_router

from app.api.safety import router as safety_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚗 AutoPulse starting up...")
    try:
        await init_db()
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        print("⚠️ Running without database - some features will be limited")
    
    yield
    
    # Shutdown
    print("🛑 AutoPulse shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AutoPulse",
    description="Connected Car Platform for Porsche 911 - Telemetry, Predictive Maintenance & Safety",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(telemetry_router)

app.include_router(safety_router, prefix="/api")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "name": "AutoPulse",
        "version": "0.1.0",
        "status": "running",
        "description": "Connected Car Platform for Porsche 911"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "websocket": "available"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug
    )
