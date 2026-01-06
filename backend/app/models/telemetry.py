"""
AutoPulse Telemetry Models
SQLAlchemy models for vehicles, telemetry readings, and trips
"""

from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import String, Integer, Float, Boolean, DateTime, SmallInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Vehicle(Base):
    """Porsche 911 vehicle entity"""
    __tablename__ = "vehicles"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    vin: Mapped[str] = mapped_column(String(17), unique=True, nullable=False)
    make: Mapped[str] = mapped_column(String(50), default="Porsche")
    model: Mapped[str] = mapped_column(String(50), default="911")
    variant: Mapped[str | None] = mapped_column(String(50))  # Carrera, Turbo S, GT3
    year: Mapped[int | None] = mapped_column(Integer)
    color: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    telemetry_readings: Mapped[list["TelemetryReading"]] = relationship(back_populates="vehicle")
    trips: Mapped[list["Trip"]] = relationship(back_populates="vehicle")
    
    # In the Vehicle class
    safety_events = relationship("SafetyEvent", back_populates="vehicle")
    safety_sessions = relationship("SafetySession", back_populates="vehicle")

    def __repr__(self) -> str:
        return f"<Vehicle {self.year} {self.make} {self.model} {self.variant}>"


class TelemetryReading(Base):
    """Single telemetry reading from vehicle sensors"""
    __tablename__ = "telemetry_readings"
    
    # TimescaleDB uses time as part of the primary key for hypertables
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, default=datetime.utcnow)
    vehicle_id: Mapped[UUID] = mapped_column(ForeignKey("vehicles.id", ondelete="CASCADE"), primary_key=True)
    
    # Core driving data
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    rpm: Mapped[int | None] = mapped_column(Integer)
    gear: Mapped[int | None] = mapped_column(SmallInteger)  # -1=R, 0=N, 1-7
    throttle_position: Mapped[float | None] = mapped_column(Float)
    
    # Engine health
    engine_temp: Mapped[float | None] = mapped_column(Float)
    oil_temp: Mapped[float | None] = mapped_column(Float)
    oil_pressure: Mapped[float | None] = mapped_column(Float)
    
    # Fuel & electrical
    fuel_level: Mapped[float | None] = mapped_column(Float)
    battery_voltage: Mapped[float | None] = mapped_column(Float)
    
    # Tire pressure (PSI)
    tire_pressure_fl: Mapped[float | None] = mapped_column(Float, default=33.0)
    tire_pressure_fr: Mapped[float | None] = mapped_column(Float, default=33.0)
    tire_pressure_rl: Mapped[float | None] = mapped_column(Float, default=32.0)
    tire_pressure_rr: Mapped[float | None] = mapped_column(Float, default=32.0)
    
    # Location
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    heading: Mapped[float | None] = mapped_column(Float, default=0)
    
    # Driving mode
    driving_mode: Mapped[str | None] = mapped_column(String(20))
    
    # ML-ready derived metrics
    acceleration_ms2: Mapped[float | None] = mapped_column(Float)
    acceleration_g: Mapped[float | None] = mapped_column(Float)
    jerk_ms3: Mapped[float | None] = mapped_column(Float)
    is_harsh_braking: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_harsh_acceleration: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_over_rpm: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_speeding: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_idling: Mapped[bool | None] = mapped_column(Boolean, default=False)
    engine_stress_score: Mapped[float | None] = mapped_column(Float)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    vehicle: Mapped["Vehicle"] = relationship(back_populates="telemetry_readings")
    
    def __repr__(self) -> str:
        return f"<TelemetryReading {self.time} speed={self.speed_kmh} rpm={self.rpm}>"


class Trip(Base):
    """A driving trip/journey with detailed analytics"""
    __tablename__ = "trips"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    vehicle_id: Mapped[UUID] = mapped_column(ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False)
    
    # Trip timing
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Core statistics (calculated on trip end)
    distance_km: Mapped[float | None] = mapped_column(Float)
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    
    # Speed stats
    avg_speed_kmh: Mapped[float | None] = mapped_column(Float)
    max_speed_kmh: Mapped[float | None] = mapped_column(Float)
    
    # RPM stats
    avg_rpm: Mapped[int | None] = mapped_column(Integer)
    max_rpm: Mapped[int | None] = mapped_column(Integer)
    
    # Fuel consumption
    fuel_start: Mapped[float | None] = mapped_column(Float)
    fuel_end: Mapped[float | None] = mapped_column(Float)
    fuel_used_liters: Mapped[float | None] = mapped_column(Float)
    
    # Mode breakdown (seconds spent in each mode)
    mode_parked_seconds: Mapped[int | None] = mapped_column(Integer, default=0)
    mode_city_seconds: Mapped[int | None] = mapped_column(Integer, default=0)
    mode_highway_seconds: Mapped[int | None] = mapped_column(Integer, default=0)
    mode_sport_seconds: Mapped[int | None] = mapped_column(Integer, default=0)
    mode_reverse_seconds: Mapped[int | None] = mapped_column(Integer, default=0)
    
    # Route endpoints
    start_latitude: Mapped[float | None] = mapped_column(Float)
    start_longitude: Mapped[float | None] = mapped_column(Float)
    end_latitude: Mapped[float | None] = mapped_column(Float)
    end_longitude: Mapped[float | None] = mapped_column(Float)
    
    # Running totals (updated in real-time during trip)
    total_readings: Mapped[int | None] = mapped_column(Integer, default=0)
    speed_sum: Mapped[float | None] = mapped_column(Float, default=0)
    rpm_sum: Mapped[float | None] = mapped_column(Float, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    vehicle: Mapped["Vehicle"] = relationship(back_populates="trips")
    
    safety_events = relationship("SafetyEvent", back_populates="trip")
    safety_session = relationship("SafetySession", back_populates="trip")
    
    
    def __repr__(self) -> str:
        status = "active" if self.is_active else "completed"
        return f"<Trip {self.id} {status} distance={self.distance_km}km>"
