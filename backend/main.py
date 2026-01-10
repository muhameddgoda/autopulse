"""
AutoPulse - Connected Car Platform
FastAPI Application Entry Point

Restructured API Routes:
- /api/telemetry/*  - Telemetry ingestion, WebSocket, ML training
- /api/vehicles/*   - Vehicle CRUD
- /api/trips/*      - Trip lifecycle
- /api/analytics/*  - Stats, exports, summaries
- /api/scoring/*    - All driver scoring (rules + ML hybrid)
- /api/safety/*     - Safety alerts
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db

# Import all routers (new modular structure)
from app.api.telemetry import router as telemetry_router
from app.api.vehicles import router as vehicles_router
from app.api.trips import router as trips_router
from app.api.analytics import router as analytics_router
from app.api.scoring import router as scoring_router
from app.api.safety import router as safety_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("[AutoPulse] Starting up...")
    try:
        await init_db()
        print("[AutoPulse] Database connected")
    except Exception as e:
        print(f"[AutoPulse] Database connection failed: {e}")
        print("[AutoPulse] Running without database - some features will be limited")
    
    yield
    
    # Shutdown
    print("[AutoPulse] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AutoPulse",
    description="Connected Car Platform for Porsche 911 - Telemetry, Predictive Maintenance & Safety",
    version="0.2.0",  # Bumped version for restructure
    lifespan=lifespan,
)

# CORS middleware - MUST be added before routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REGISTER ROUTERS (New Modular Structure)
# ============================================

# Telemetry: /api/telemetry/* 
# - POST /readings
# - GET /readings/{vehicle_id}
# - WS /ws/{vehicle_id}
# - GET/POST /ml/* (model status, training)
app.include_router(telemetry_router, prefix="/api/telemetry", tags=["telemetry"])

# Vehicles: /api/vehicles/*
# - POST / (create)
# - GET / (list)
# - GET /{id} (get one)
# - DELETE /{id}
app.include_router(vehicles_router, prefix="/api/vehicles", tags=["vehicles"])

# Trips: /api/trips/*
# - POST /start/{vehicle_id}
# - POST /end/{trip_id}
# - GET /{vehicle_id}
# - GET /active/{vehicle_id}
# - GET /details/{trip_id}
app.include_router(trips_router, prefix="/api/trips", tags=["trips"])

# Analytics: /api/analytics/*
# - GET /stats/weekly/{vehicle_id}
# - GET /stats/daily/{vehicle_id}
# - GET /export/telemetry/{vehicle_id}
# - GET /export/trips/{vehicle_id}
# - GET /summary/{vehicle_id}
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])

# Scoring: /api/scoring/* (already has prefix in router)
# - POST /trips/{trip_id}/calculate
# - POST /trips/{trip_id}/hybrid
# - GET /trips/{vehicle_id}
# - GET /trips/{trip_id}/details
# - GET /trips/{trip_id}/timeline
# - GET /history/{vehicle_id}
# - GET /summary/{vehicle_id}
# - POST /backfill/{vehicle_id}
app.include_router(scoring_router, tags=["scoring"])

# Safety: /api/safety/*
app.include_router(safety_router, prefix="/api/safety", tags=["safety"])

# ============================================
# HEALTH ENDPOINTS
# ============================================


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "name": "AutoPulse",
        "version": "0.2.0",
        "status": "running",
        "description": "Connected Car Platform for Porsche 911",
        "api_docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "websocket": "available",
        "api_routes": {
            "telemetry": "/api/telemetry",
            "vehicles": "/api/vehicles",
            "trips": "/api/trips",
            "analytics": "/api/analytics",
            "scoring": "/api/scoring",
            "safety": "/api/safety"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug
    )
