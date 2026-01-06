"""
AutoPulse Safety Models
=======================
SQLAlchemy models for storing safety events and monitoring sessions.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class SafetyEvent(Base):
    """
    Safety event record.
    
    Stores drowsiness, distraction, and other safety-related events
    detected during driving.
    """
    __tablename__ = "safety_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vehicle_id = Column(UUID(as_uuid=True), ForeignKey("vehicles.id"), nullable=False, index=True)
    trip_id = Column(UUID(as_uuid=True), ForeignKey("trips.id"), nullable=True, index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)
    # Types: drowsiness_warning, drowsiness_alert, drowsiness_critical,
    #        distraction_warning, distraction_alert, high_blink_rate,
    #        face_not_detected
    
    severity = Column(String(20), nullable=False, index=True)
    # Severities: info, warning, critical
    
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    duration_seconds = Column(Float, default=0.0)
    
    # Additional event data
    details = Column(JSONB, default={})
    # Example: {"ear_average": 0.18, "blinks_per_minute": 28, "head_pose": {...}}
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utc_now)
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="safety_events")
    trip = relationship("Trip", back_populates="safety_events")
    
    def __repr__(self):
        return f"<SafetyEvent {self.event_type} severity={self.severity}>"


class SafetySession(Base):
    """
    Safety monitoring session.
    
    Tracks a continuous period of driver safety monitoring,
    storing aggregate statistics.
    """
    __tablename__ = "safety_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vehicle_id = Column(UUID(as_uuid=True), ForeignKey("vehicles.id"), nullable=False, index=True)
    trip_id = Column(UUID(as_uuid=True), ForeignKey("trips.id"), nullable=True, index=True)
    
    # Session timing
    start_time = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    end_time = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # Aggregate statistics
    total_drowsy_seconds = Column(Float, default=0.0)
    total_distracted_seconds = Column(Float, default=0.0)
    drowsiness_events = Column(Integer, default=0)
    distraction_events = Column(Integer, default=0)
    average_safety_score = Column(Float, default=100.0)
    
    # Processing stats
    frames_processed = Column(Integer, default=0)
    average_fps = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="safety_sessions")
    trip = relationship("Trip", back_populates="safety_session")
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (utc_now() - self.start_time).total_seconds()
    
    @property
    def drowsy_percentage(self) -> float:
        """Percentage of time spent drowsy"""
        duration = self.duration_seconds
        if duration > 0:
            return (self.total_drowsy_seconds / duration) * 100
        return 0.0
    
    @property
    def distracted_percentage(self) -> float:
        """Percentage of time spent distracted"""
        duration = self.duration_seconds
        if duration > 0:
            return (self.total_distracted_seconds / duration) * 100
        return 0.0
    
    def __repr__(self):
        status = "active" if self.is_active else "ended"
        return f"<SafetySession {self.id} {status} score={self.average_safety_score}>"
