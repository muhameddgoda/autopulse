"""
AutoPulse Safety Schemas
========================
Pydantic schemas for safety API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from uuid import UUID


# ============================================
# DETECTION SCHEMAS
# ============================================

class DetectionRequest(BaseModel):
    """Request for single-frame drowsiness detection"""
    image_base64: str = Field(..., description="Base64-encoded image data")
    vehicle_id: Optional[UUID] = Field(None, description="Vehicle ID for event storage")
    trip_id: Optional[UUID] = Field(None, description="Trip ID for event association")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
                "vehicle_id": "68f2ce4a-28df-4f11-bf5b-c961d1f7d064"
            }
        }


class DetectionResponse(BaseModel):
    """Response from drowsiness detection"""
    success: bool
    state: Dict[str, Any]
    message: str = ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "state": {
                    "drowsiness": {
                        "ear_left": 0.28,
                        "ear_right": 0.27,
                        "ear_average": 0.275,
                        "eyes_closed": False,
                        "closed_duration_ms": 0,
                        "blink_count": 5,
                        "blinks_per_minute": 12.5,
                        "alert_level": "none",
                        "is_drowsy": False,
                        "face_detected": True
                    },
                    "is_safe": True,
                    "safety_score": 95.0,
                    "active_alerts": []
                },
                "message": "Detection successful"
            }
        }


# ============================================
# SESSION SCHEMAS
# ============================================

class SessionStartRequest(BaseModel):
    """Request to start a safety monitoring session"""
    vehicle_id: UUID
    trip_id: Optional[UUID] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": "68f2ce4a-28df-4f11-bf5b-c961d1f7d064"
            }
        }


class SessionEndRequest(BaseModel):
    """Request to end a safety monitoring session"""
    pass  # No additional data needed


class SafetySessionResponse(BaseModel):
    """Response for session operations"""
    id: UUID
    vehicle_id: UUID
    trip_id: Optional[UUID] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    is_active: bool
    total_drowsy_seconds: float = 0.0
    total_distracted_seconds: float = 0.0
    drowsiness_events: int = 0
    distraction_events: int = 0
    average_safety_score: float = 100.0
    message: str = ""
    
    class Config:
        from_attributes = True


# ============================================
# EVENT SCHEMAS
# ============================================

class SafetyEventResponse(BaseModel):
    """Safety event record"""
    id: UUID
    vehicle_id: UUID
    trip_id: Optional[UUID] = None
    event_type: str
    severity: str
    timestamp: datetime
    duration_seconds: float
    details: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "vehicle_id": "68f2ce4a-28df-4f11-bf5b-c961d1f7d064",
                "event_type": "drowsiness_alert",
                "severity": "warning",
                "timestamp": "2026-01-05T10:30:00Z",
                "duration_seconds": 2.5,
                "details": {
                    "ear_average": 0.18,
                    "blinks_per_minute": 28
                }
            }
        }


class SafetyEventCreate(BaseModel):
    """Schema for creating a safety event"""
    vehicle_id: UUID
    trip_id: Optional[UUID] = None
    event_type: str
    severity: str
    duration_seconds: float = 0.0
    details: Dict[str, Any] = {}


# ============================================
# SUMMARY SCHEMAS
# ============================================

class SafetySummaryResponse(BaseModel):
    """Safety statistics summary"""
    vehicle_id: UUID
    period_days: int
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    total_drowsy_seconds: float
    total_distracted_seconds: float
    average_safety_score: float
    session_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": "68f2ce4a-28df-4f11-bf5b-c961d1f7d064",
                "period_days": 30,
                "total_events": 15,
                "events_by_type": {
                    "drowsiness_warning": 8,
                    "drowsiness_alert": 4,
                    "distraction_warning": 3
                },
                "events_by_severity": {
                    "info": 2,
                    "warning": 10,
                    "critical": 3
                },
                "total_drowsy_seconds": 45.5,
                "total_distracted_seconds": 30.2,
                "average_safety_score": 85.5,
                "session_count": 25
            }
        }


# ============================================
# WEBSOCKET MESSAGE SCHEMAS
# ============================================

class WebSocketFrameMessage(BaseModel):
    """WebSocket message for sending a frame"""
    type: str = "frame"
    data: str  # Base64 encoded image
    
    
class WebSocketDetectionMessage(BaseModel):
    """WebSocket message with detection results"""
    type: str = "detection"
    data: Dict[str, Any]
    frame: int
    

class WebSocketAlertMessage(BaseModel):
    """WebSocket message for alerts"""
    type: str = "alert"
    data: Dict[str, Any]
