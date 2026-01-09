"""
AutoPulse Safety API
====================
REST and WebSocket endpoints for driver safety monitoring.

Endpoints:
- POST /api/safety/detect - Single frame detection
- WS /api/safety/stream - Real-time WebSocket streaming
- GET /api/safety/events/{vehicle_id} - Get safety events
- GET /api/safety/summary/{vehicle_id} - Get safety summary
- POST /api/safety/session/start - Start monitoring session
- POST /api/safety/session/end - End monitoring session
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from uuid import UUID
import json
import asyncio
import logging
import base64

from app.database import get_db
from app.models.telemetry import Vehicle, Trip
from app.models.safety import SafetyEvent as SafetyEventModel, SafetySession
from app.schemas.safety import (
    DetectionRequest, DetectionResponse,
    SafetyEventResponse, SafetySessionResponse,
    SafetySummaryResponse, SessionStartRequest, SessionEndRequest
)

# Import CV modules (with fallback if not available)
try:
    from app.cv.eye_detector import (
        get_eye_detector,
        EyeDrowsinessDetector,
        is_eye_detector_available
    )
    CV_AVAILABLE = is_eye_detector_available()
except ImportError:
    CV_AVAILABLE = False
    get_eye_detector = None
    EyeDrowsinessDetector = None

# Fallback imports for backward compatibility
try:
    from app.cv import (
        get_drowsiness_detector,
        DriverSafetyState,
        SafetyEvent,
        MEDIAPIPE_AVAILABLE
    )
except ImportError:
    get_drowsiness_detector = None
    DriverSafetyState = None
    SafetyEvent = None
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(tags=["safety"])

# Global state for current drowsiness status (per vehicle)
_current_drowsiness_status: dict = {}

# ============================================
# STATUS ENDPOINT (for HUD polling)
# ============================================


@router.get("/status/{vehicle_id}")
async def get_drowsiness_status(vehicle_id: UUID):
    """
    Get current drowsiness status for a vehicle.
    Used by HUD to display warnings.
    """
    status = _current_drowsiness_status.get(str(vehicle_id), {})
    return {
        "alert_level": status.get("alert_level", "none"),
        "is_drowsy": status.get("is_drowsy", False),
        "confidence": status.get("confidence", 0),
        "timestamp": status.get("timestamp", None)
    }

# ============================================
# DETECTION ENDPOINTS
# ============================================


@router.post("/detect", response_model=DetectionResponse)
async def detect_drowsiness(
    request: DetectionRequest,
    db: AsyncSession=Depends(get_db)
):
    """
    Detect drowsiness from a single image frame.
    
    Send a base64-encoded image and get back drowsiness detection results.
    For real-time detection, use the WebSocket endpoint instead.
    """
    if not CV_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Computer vision module not available. Install mediapipe, opencv-python, and torch."
        )
    
    detector = get_eye_detector(model_path="models/eye_classifier.pth")
    
    try:
        # Process the frame
        state = detector.process_base64(request.image_base64)
        
        # Build response in format frontend expects
        response_data = {
            "drowsiness": state.to_dict(),
            "is_safe": not state.is_drowsy,
            "safety_score": 100 - (state.closed_duration_ms / 40),  # Simple score calc
            "active_alerts": ["drowsiness"] if state.is_drowsy else [],
            "session_stats": detector.get_session_summary()
        }
        
        return DetectionResponse(
            success=True,
            state=response_data,
            message="Detection successful"
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/stream/{vehicle_id}")
async def drowsiness_stream(
    websocket: WebSocket,
    vehicle_id: UUID,
    trip_id: Optional[UUID]=None
):
    """
    WebSocket endpoint for real-time drowsiness detection.
    
    Client sends base64-encoded image frames.
    Server responds with detection results for each frame.
    
    Message format (client -> server):
    {
        "type": "frame",
        "data": "<base64 encoded image>"
    }
    
    Message format (server -> client):
    {
        "type": "detection",
        "data": { ... detection results ... }
    }
    """
    if not CV_AVAILABLE:
        await websocket.close(code=1003, reason="CV module not available")
        return
    
    await websocket.accept()
    logger.info(f"Safety WebSocket connected for vehicle {vehicle_id}")
    
    # Use the eye detector
    detector = get_eye_detector(model_path="models/eye_classifier.pth")
    detector.reset()  # Fresh session
    
    frame_count = 0
    last_alert_level = "none"
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_json()
            
            if message.get("type") == "frame":
                frame_data = message.get("data", "")
                
                if frame_data:
                    try:
                        # Process frame
                        state = detector.process_base64(frame_data)
                        frame_count += 1
                        
                        # Build response in format frontend expects
                        session_stats = detector.get_session_summary()
                        
                        response_data = {
                            "drowsiness": state.to_dict(),
                            "is_safe": not state.is_drowsy,
                            "safety_score": max(0, 100 - (state.closed_duration_ms / 40)),
                            "active_alerts": ["drowsiness"] if state.is_drowsy else [],
                            "session_stats": session_stats
                        }
                        
                        # Update global status for HUD polling
                        _current_drowsiness_status[str(vehicle_id)] = {
                            "alert_level": state.alert_level.value,
                            "is_drowsy": state.is_drowsy,
                            "confidence": state.confidence,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Send detection result
                        await websocket.send_json({
                            "type": "detection",
                            "data": response_data,
                            "frame": frame_count
                        })
                        
                        # Send alert if level changed to alert/critical
                        current_alert = state.alert_level.value
                        if current_alert in ["alert", "critical"] and current_alert != last_alert_level:
                            await websocket.send_json({
                                "type": "alert",
                                "data": {
                                    "type": "drowsiness_detected",
                                    "severity": current_alert,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "duration_seconds": state.closed_duration_ms / 1000,
                                    "details": {
                                        "ear_average": state.ear_average,
                                        "confidence": state.confidence
                                    }
                                }
                            })
                        last_alert_level = current_alert
                        
                    except Exception as e:
                        logger.error(f"Frame processing error: {e}")
                        # Send error but keep connection alive
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                    
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
            elif message.get("type") == "end":
                # Client wants to end session
                break
                
    except WebSocketDisconnect:
        logger.info(f"Safety WebSocket disconnected for vehicle {vehicle_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clear status when disconnected
        _current_drowsiness_status.pop(str(vehicle_id), None)
        # Log session summary
        try:
            summary = detector.get_session_summary()
            logger.info(f"Session ended: {summary}")
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")


@router.get("/status")
async def get_safety_status():
    """Check if the safety/CV module is available"""
    detector_info = "Not loaded"
    
    if CV_AVAILABLE:
        try:
            detector = get_eye_detector(model_path="models/eye_classifier.pth")
            detector_info = f"Eye classifier loaded, enabled={detector.enabled}"
        except Exception as e:
            detector_info = f"Error: {e}"
    
    return {
        "cv_available": CV_AVAILABLE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE if 'MEDIAPIPE_AVAILABLE' in dir() else False,
        "detector": detector_info,
        "message": "Ready for drowsiness detection" if CV_AVAILABLE else "Install dependencies: pip install mediapipe opencv-python torch torchvision"
    }

# ============================================
# SESSION MANAGEMENT
# ============================================


@router.post("/session/start", response_model=SafetySessionResponse)
async def start_safety_session(
    request: SessionStartRequest,
    db: AsyncSession=Depends(get_db)
):
    """
    Start a new safety monitoring session.
    """
    # Verify vehicle exists
    result = await db.execute(
        select(Vehicle).where(Vehicle.id == request.vehicle_id)
    )
    vehicle = result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Create session
    session = SafetySession(
        vehicle_id=request.vehicle_id,
        trip_id=request.trip_id,
        start_time=datetime.now(timezone.utc),
        is_active=True
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    # Reset detector if available
    if CV_AVAILABLE:
        detector = get_eye_detector(model_path="models/eye_classifier.pth")
        detector.reset()
    
    return SafetySessionResponse(
        id=session.id,
        vehicle_id=session.vehicle_id,
        trip_id=session.trip_id,
        start_time=session.start_time,
        is_active=True,
        message="Safety monitoring session started"
    )


@router.post("/session/{session_id}/end", response_model=SafetySessionResponse)
async def end_safety_session(
    session_id: UUID,
    request: SessionEndRequest,
    db: AsyncSession=Depends(get_db)
):
    """End a safety monitoring session and save statistics."""
    result = await db.execute(
        select(SafetySession).where(SafetySession.id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session already ended")
    
    # Get session summary from detector
    summary = {}
    if CV_AVAILABLE:
        detector = get_eye_detector(model_path="models/eye_classifier.pth")
        summary = detector.get_session_summary()
    
    # Update session
    session.end_time = datetime.now(timezone.utc)
    session.is_active = False
    session.total_drowsy_seconds = summary.get("total_drowsy_seconds", 0)
    session.total_distracted_seconds = summary.get("total_distracted_seconds", 0)
    session.drowsiness_events = summary.get("drowsiness_events", 0)
    session.distraction_events = summary.get("distraction_events", 0)
    session.average_safety_score = summary.get("average_safety_score", 100)
    
    await db.commit()
    await db.refresh(session)
    
    return SafetySessionResponse(
        id=session.id,
        vehicle_id=session.vehicle_id,
        trip_id=session.trip_id,
        start_time=session.start_time,
        end_time=session.end_time,
        is_active=False,
        total_drowsy_seconds=session.total_drowsy_seconds,
        total_distracted_seconds=session.total_distracted_seconds,
        drowsiness_events=session.drowsiness_events,
        distraction_events=session.distraction_events,
        average_safety_score=session.average_safety_score,
        message="Safety monitoring session ended"
    )

# ============================================
# EVENTS & HISTORY
# ============================================


@router.get("/events/{vehicle_id}", response_model=List[SafetyEventResponse])
async def get_safety_events(
    vehicle_id: UUID,
    trip_id: Optional[UUID]=None,
    severity: Optional[str]=None,
    limit: int=Query(default=50, ge=1, le=200),
    days: int=Query(default=7, ge=1, le=90),
    db: AsyncSession=Depends(get_db)
):
    """Get safety events for a vehicle."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    
    query = select(SafetyEventModel).where(
        SafetyEventModel.vehicle_id == vehicle_id,
        SafetyEventModel.timestamp >= since
    )
    
    if trip_id:
        query = query.where(SafetyEventModel.trip_id == trip_id)
    
    if severity:
        query = query.where(SafetyEventModel.severity == severity)
    
    query = query.order_by(desc(SafetyEventModel.timestamp)).limit(limit)
    
    result = await db.execute(query)
    events = result.scalars().all()
    
    return [
        SafetyEventResponse(
            id=e.id,
            vehicle_id=e.vehicle_id,
            trip_id=e.trip_id,
            event_type=e.event_type,
            severity=e.severity,
            timestamp=e.timestamp,
            duration_seconds=e.duration_seconds,
            details=e.details or {}
        )
        for e in events
    ]


@router.get("/summary/{vehicle_id}", response_model=SafetySummaryResponse)
async def get_safety_summary(
    vehicle_id: UUID,
    days: int=Query(default=30, ge=1, le=365),
    db: AsyncSession=Depends(get_db)
):
    """Get safety summary statistics for a vehicle."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Count events by type
    result = await db.execute(
        select(
            SafetyEventModel.event_type,
            func.count(SafetyEventModel.id).label("count")
        )
        .where(SafetyEventModel.vehicle_id == vehicle_id)
        .where(SafetyEventModel.timestamp >= since)
        .group_by(SafetyEventModel.event_type)
    )
    event_counts = {row.event_type: row.count for row in result}
    
    # Count events by severity
    result = await db.execute(
        select(
            SafetyEventModel.severity,
            func.count(SafetyEventModel.id).label("count")
        )
        .where(SafetyEventModel.vehicle_id == vehicle_id)
        .where(SafetyEventModel.timestamp >= since)
        .group_by(SafetyEventModel.severity)
    )
    severity_counts = {row.severity: row.count for row in result}
    
    # Get session statistics
    result = await db.execute(
        select(
            func.sum(SafetySession.total_drowsy_seconds),
            func.sum(SafetySession.total_distracted_seconds),
            func.avg(SafetySession.average_safety_score),
            func.count(SafetySession.id)
        )
        .where(SafetySession.vehicle_id == vehicle_id)
        .where(SafetySession.start_time >= since)
    )
    row = result.one()
    
    total_drowsy = row[0] or 0
    total_distracted = row[1] or 0
    avg_score = row[2] or 100
    session_count = row[3] or 0
    
    return SafetySummaryResponse(
        vehicle_id=vehicle_id,
        period_days=days,
        total_events=sum(event_counts.values()) if event_counts else 0,
        events_by_type=event_counts,
        events_by_severity=severity_counts,
        total_drowsy_seconds=total_drowsy,
        total_distracted_seconds=total_distracted,
        average_safety_score=avg_score,
        session_count=session_count
    )
