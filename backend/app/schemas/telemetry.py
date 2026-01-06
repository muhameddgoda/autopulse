"""
AutoPulse Telemetry Schemas (FIXED)
Pydantic models for API request/response validation

FIXES:
- Added optional 'timestamp' field to TelemetryReadingCreate for training data
- Added optional 'end_timestamp' field to TripEnd for simulated trips
"""

from datetime import datetime
from uuid import UUID
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
    battery_voltage: float = Field(ge=0, le=20, description="Battery voltage V")
    
    # Tire pressure (PSI)
    tire_pressure_fl: float | None = Field(default=33.0, ge=0, le=60, description="Front Left PSI")
    tire_pressure_fr: float | None = Field(default=33.0, ge=0, le=60, description="Front Right PSI")
    tire_pressure_rl: float | None = Field(default=32.0, ge=0, le=60, description="Rear Left PSI")
    tire_pressure_rr: float | None = Field(default=32.0, ge=0, le=60, description="Rear Right PSI")
    
    # Location
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    heading: float | None = Field(default=None, ge=0, le=360, description="Heading in degrees")
    
    # Driving mode (optional, sent by simulator)
    driving_mode: str | None = Field(default=None, description="city, highway, sport")
    
    # ML-ready derived metrics
    acceleration_ms2: float | None = Field(default=None, description="Acceleration m/s²")
    acceleration_g: float | None = Field(default=None, description="Acceleration in G-force")
    jerk_ms3: float | None = Field(default=None, description="Rate of acceleration change")
    is_harsh_braking: bool | None = Field(default=False, description="Currently harsh braking")
    is_harsh_acceleration: bool | None = Field(default=False, description="Currently harsh accelerating")
    is_over_rpm: bool | None = Field(default=False, description="RPM above threshold")
    is_speeding: bool | None = Field(default=False, description="Speed above threshold")
    is_idling: bool | None = Field(default=False, description="Engine on but not moving")
    engine_stress_score: float | None = Field(default=None, ge=0, le=100, description="Engine stress 0-100")


class TelemetryReadingCreate(TelemetryReadingBase):
    """Create a new telemetry reading"""
    vehicle_id: UUID
    # FIXED: Optional timestamp for training data generation
    # If not provided, server will use current time
    timestamp: datetime | str | None = Field(
        default=None, 
        description="Optional timestamp for the reading (ISO format). If not provided, server uses current time."
    )


class TelemetryReadingResponse(TelemetryReadingBase):
    """Telemetry reading response with timestamp"""
    time: datetime
    vehicle_id: UUID
    
    model_config = ConfigDict(from_attributes=True)


class TelemetryStreamMessage(TelemetryReadingBase):
    """WebSocket message format for real-time streaming"""
    timestamp: datetime
    vehicle_id: UUID
    
    # Computed fields for dashboard
    engine_status: str = "normal"  # normal, warning, critical
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# TRIP SCHEMAS
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
    # FIXED: Optional end timestamp for simulated/training data
    end_timestamp: datetime | str | None = Field(
        default=None,
        description="Optional end timestamp for simulated trips (ISO format). If not provided, server calculates from readings."
    )


class TripResponse(BaseModel):
    """Trip response with all details including mode breakdown"""
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
    
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# DASHBOARD SCHEMAS
# ============================================

class DashboardState(BaseModel):
    """Complete dashboard state for frontend"""
    vehicle: VehicleResponse
    telemetry: TelemetryReadingResponse | None
    active_trip: TripResponse | None
    connection_status: str = "connected"