"""
AutoPulse Telemetry Schemas (Updated with Scoring)
Pydantic models for API request/response validation

UPDATED: Added driver scoring fields to TripResponse
"""

from datetime import datetime
from uuid import UUID
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

# ============================================
# VEHICLE SCHEMAS
# ============================================


class VehicleBase(BaseModel):
    """Base vehicle data"""
    vin: str = Field(..., min_length=17, max_length=17)
    make: str = "Porsche"
    model: str = "911"
    variant: str | None = None
    year: int | None = None
    color: str | None = None


class VehicleCreate(VehicleBase):
    """Create a new vehicle"""
    pass


class VehicleResponse(VehicleBase):
    """Vehicle response with ID and timestamps"""
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# ============================================
# TELEMETRY SCHEMAS
# ============================================


class TelemetryReadingBase(BaseModel):
    """Base telemetry reading data"""
    # Core driving
    speed_kmh: float = Field(ge=0, le=400, description="Speed in km/h")
    rpm: int = Field(ge=0, le=10000, description="Engine RPM")
    gear: int = Field(ge=-1, le=7, description="-1=R, 0=N, 1-7 forward")
    throttle_position: float = Field(ge=0, le=100, description="Throttle %")
    
    # Engine health
    engine_temp: float = Field(ge=-40, le=150, description="Engine temp °C")
    oil_temp: float = Field(ge=-40, le=180, description="Oil temp °C")
    oil_pressure: float = Field(ge=0, le=10, description="Oil pressure bar")
    
    # Fuel & electrical
    fuel_level: float = Field(ge=0, le=100, description="Fuel level %")
    battery_voltage: float | None = Field(default=12.6, ge=0, le=20, description="Battery voltage V")
    
    # Tire pressure (PSI)
    tire_pressure_fl: float | None = Field(default=33.0, ge=0, le=60)
    tire_pressure_fr: float | None = Field(default=33.0, ge=0, le=60)
    tire_pressure_rl: float | None = Field(default=32.0, ge=0, le=60)
    tire_pressure_rr: float | None = Field(default=32.0, ge=0, le=60)
    
    # Location
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    heading: float | None = Field(default=None, ge=0, le=360)
    
    # Driving mode
    driving_mode: str | None = Field(default=None)
    
    # ML-ready derived metrics
    acceleration_ms2: float | None = None
    acceleration_g: float | None = None
    jerk_ms3: float | None = None
    is_harsh_braking: bool | None = False
    is_harsh_acceleration: bool | None = False
    is_over_rpm: bool | None = False
    is_speeding: bool | None = False
    is_idling: bool | None = False
    engine_stress_score: float | None = None


class TelemetryReadingCreate(TelemetryReadingBase):
    """Create a new telemetry reading"""
    vehicle_id: UUID
    time: datetime | str | None = Field(default=None)


class TelemetryBatchCreate(BaseModel):
    """Batch create telemetry readings"""
    readings: List["TelemetryReadingCreate"]


class TelemetryReadingResponse(TelemetryReadingBase):
    """Telemetry reading response with timestamp"""
    time: datetime
    vehicle_id: UUID
    
    model_config = ConfigDict(from_attributes=True)

# ============================================
# TRIP SCHEMAS (UPDATED WITH SCORING)
# ============================================


class TripBase(BaseModel):
    """Base trip data"""
    vehicle_id: UUID


class TripCreate(TripBase):
    """Start a new trip"""
    start_latitude: float | None = None
    start_longitude: float | None = None


class TripEnd(BaseModel):
    """End an active trip"""
    end_latitude: float | None = None
    end_longitude: float | None = None
    end_timestamp: datetime | str | None = None


class TripResponse(BaseModel):
    """Trip response with all details including driver scoring"""
    id: UUID
    vehicle_id: UUID
    start_time: datetime
    end_time: datetime | None
    is_active: bool
    
    # Core statistics
    distance_km: float | None
    duration_seconds: int | None
    
    # Speed stats
    avg_speed_kmh: float | None
    max_speed_kmh: float | None
    
    # RPM stats
    avg_rpm: int | None
    max_rpm: int | None
    
    # Fuel stats
    fuel_start: float | None
    fuel_end: float | None
    fuel_used_liters: float | None
    
    # Mode breakdown (seconds)
    mode_parked_seconds: int | None
    mode_city_seconds: int | None
    mode_highway_seconds: int | None
    mode_sport_seconds: int | None
    mode_reverse_seconds: int | None
    
    # Route
    start_latitude: float | None
    start_longitude: float | None
    end_latitude: float | None
    end_longitude: float | None
    
    # ========================================
    # NEW: Driver Behavior Scoring
    # ========================================
    driver_score: float | None = Field(None, description="Driver behavior score 0-100")
    behavior_label: str | None = Field(None, description="exemplary, calm, normal, aggressive, dangerous")
    risk_level: str | None = Field(None, description="low, medium, high, critical")
    harsh_brake_count: int | None = Field(0, description="Number of harsh braking events")
    harsh_accel_count: int | None = Field(0, description="Number of harsh acceleration events")
    speeding_percentage: float | None = Field(0, description="Percentage of trip spent speeding")
    ml_enhanced: bool | None = Field(False, description="Whether ML model was used for scoring")
    # ========================================
    
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class TripWithScoreResponse(BaseModel):
    """Simplified trip response for sidebar/list display"""
    id: str
    vehicle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    is_active: bool
    
    # Core stats
    distance_km: Optional[float]
    duration_seconds: Optional[int]
    
    # Scoring (main focus)
    driver_score: Optional[float]
    behavior_label: Optional[str]
    risk_level: Optional[str]
    
    model_config = ConfigDict(from_attributes=True)


class TripScoreDetailResponse(BaseModel):
    """Detailed scoring response with insights"""
    trip_id: str
    vehicle_id: str
    
    # Trip info
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: Optional[int]
    distance_km: Optional[float]
    
    # Score
    driver_score: Optional[float]
    behavior_label: Optional[str]
    risk_level: Optional[str]
    ml_enhanced: Optional[bool]
    
    # Events
    harsh_brake_count: int
    harsh_accel_count: int
    speeding_percentage: float
    
    # Stats
    avg_speed_kmh: Optional[float]
    max_speed_kmh: Optional[float]
    
    # Analysis
    summary: dict
    insights: List[str]
    recommendations: List[str]


class ScoreHistoryItem(BaseModel):
    """Single score history entry"""
    trip_id: str
    date: str
    score: float
    behavior: str
    distance_km: Optional[float]


class ScoreHistoryResponse(BaseModel):
    """Score history for charts"""
    vehicle_id: str
    history: List[ScoreHistoryItem]
    count: int

# ============================================
# DASHBOARD SCHEMAS
# ============================================


class DashboardState(BaseModel):
    """Complete dashboard state for frontend"""
    vehicle: VehicleResponse
    telemetry: TelemetryReadingResponse | None
    active_trip: TripResponse | None
    connection_status: str = "connected"


class TelemetryStreamMessage(TelemetryReadingBase):
    """WebSocket message format for real-time streaming"""
    timestamp: datetime
    vehicle_id: UUID
    
    # Computed fields for dashboard
    engine_status: str = "normal"  # normal, warning, critical
    
    model_config = ConfigDict(from_attributes=True)
