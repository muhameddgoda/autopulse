"""
AutoPulse Drowsiness Detector Service
======================================
High-level service combining eye tracking, head pose, and alert management.

This is the main interface for the drowsiness detection feature.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time
import asyncio
import logging

from app.cv.eye_tracker import (
    EyeTracker, DrowsinessState, AlertLevel, 
    MEDIAPIPE_AVAILABLE, get_eye_tracker
)
from app.cv.head_pose import HeadPoseEstimator, DistractionState

logger = logging.getLogger(__name__)


class SafetyEventType(Enum):
    """Types of safety events"""
    DROWSINESS_WARNING = "drowsiness_warning"
    DROWSINESS_ALERT = "drowsiness_alert"
    DROWSINESS_CRITICAL = "drowsiness_critical"
    DISTRACTION_WARNING = "distraction_warning"
    DISTRACTION_ALERT = "distraction_alert"
    HIGH_BLINK_RATE = "high_blink_rate"
    FACE_NOT_DETECTED = "face_not_detected"


@dataclass
class SafetyEvent:
    """A safety event to be stored"""
    event_type: SafetyEventType
    severity: str  # "info", "warning", "critical"
    timestamp: datetime
    duration_seconds: float
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "details": self.details
        }


@dataclass
class DriverSafetyState:
    """Complete driver safety state"""
    # Drowsiness
    drowsiness: DrowsinessState
    
    # Distraction (optional, if head pose available)
    distraction: Optional[DistractionState] = None
    
    # Overall safety
    is_safe: bool = True
    safety_score: float = 100.0
    active_alerts: List[str] = field(default_factory=list)
    
    # Session stats
    session_start: float = 0.0
    total_drowsy_seconds: float = 0.0
    total_distracted_seconds: float = 0.0
    drowsiness_events: int = 0
    distraction_events: int = 0
    
    def to_dict(self) -> Dict:
        result = {
            "drowsiness": self.drowsiness.to_dict(),
            "is_safe": self.is_safe,
            "safety_score": round(self.safety_score, 1),
            "active_alerts": self.active_alerts,
            "session_stats": {
                "duration_seconds": round(time.time() - self.session_start, 1) if self.session_start else 0,
                "total_drowsy_seconds": round(self.total_drowsy_seconds, 1),
                "total_distracted_seconds": round(self.total_distracted_seconds, 1),
                "drowsiness_events": self.drowsiness_events,
                "distraction_events": self.distraction_events
            }
        }
        if self.distraction:
            result["distraction"] = self.distraction.to_dict()
        return result


class DrowsinessDetector:
    """
    Main drowsiness detection service.
    
    Combines eye tracking and head pose estimation to provide
    comprehensive driver safety monitoring.
    
    Usage:
        detector = DrowsinessDetector()
        
        # Process frames
        state = detector.process_frame(frame)
        
        # Or process base64 images (from WebSocket)
        state = detector.process_base64(base64_data)
        
        # Get current state
        current = detector.get_state()
        
        # Register alert callback
        detector.on_alert(my_callback)
    """
    
    def __init__(self, enable_head_pose: bool = True):
        """
        Initialize the drowsiness detector.
        
        Args:
            enable_head_pose: Whether to enable head pose estimation
        """
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available. Drowsiness detection disabled.")
            self._enabled = False
            return
        
        self._enabled = True
        self._eye_tracker = EyeTracker()
        self._head_pose_estimator = HeadPoseEstimator() if enable_head_pose else None
        
        # State tracking
        self._session_start = time.time()
        self._total_drowsy_seconds = 0.0
        self._total_distracted_seconds = 0.0
        self._drowsiness_events = 0
        self._distraction_events = 0
        self._last_drowsy_state = False
        self._last_distracted_state = False
        self._last_update = time.time()
        
        # Current state
        self._current_state: Optional[DriverSafetyState] = None
        
        # Event callbacks
        self._alert_callbacks: List[Callable[[SafetyEvent], None]] = []
        
        # Event history for current session
        self._events: List[SafetyEvent] = []
        
        logger.info("DrowsinessDetector initialized")
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def process_frame(self, frame: np.ndarray) -> DriverSafetyState:
        """
        Process a video frame and return safety state.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            DriverSafetyState with all safety metrics
        """
        if not self._enabled:
            return self._disabled_state()
        
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        # Get drowsiness state
        drowsiness = self._eye_tracker.process_frame(frame)
        
        # Get distraction state (if enabled)
        distraction = None
        if self._head_pose_estimator and drowsiness.face_detected:
            # We need the landmarks for head pose, but EyeTracker doesn't expose them
            # So we use the head pose values from drowsiness state
            if drowsiness.head_pitch is not None:
                from app.cv.head_pose import HeadPose
                head_pose = HeadPose(
                    pitch=drowsiness.head_pitch,
                    yaw=drowsiness.head_yaw or 0,
                    roll=drowsiness.head_roll or 0
                )
                distraction = self._head_pose_estimator.detect_distraction(head_pose)
        
        # Update statistics
        self._update_stats(drowsiness, distraction, dt)
        
        # Generate events
        self._check_and_emit_events(drowsiness, distraction)
        
        # Calculate overall safety
        is_safe, safety_score, alerts = self._calculate_safety(drowsiness, distraction)
        
        # Build state
        state = DriverSafetyState(
            drowsiness=drowsiness,
            distraction=distraction,
            is_safe=is_safe,
            safety_score=safety_score,
            active_alerts=alerts,
            session_start=self._session_start,
            total_drowsy_seconds=self._total_drowsy_seconds,
            total_distracted_seconds=self._total_distracted_seconds,
            drowsiness_events=self._drowsiness_events,
            distraction_events=self._distraction_events
        )
        
        self._current_state = state
        return state
    
    def process_base64(self, base64_data: str) -> DriverSafetyState:
        """
        Process a base64-encoded image frame.
        
        Args:
            base64_data: Base64 encoded image
            
        Returns:
            DriverSafetyState
        """
        if not self._enabled:
            return self._disabled_state()
        
        drowsiness = self._eye_tracker.process_base64_frame(base64_data)
        
        # Simplified processing for base64 (no head pose re-estimation)
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        # Get distraction from drowsiness head pose
        distraction = None
        if self._head_pose_estimator and drowsiness.face_detected:
            if drowsiness.head_pitch is not None:
                from app.cv.head_pose import HeadPose
                head_pose = HeadPose(
                    pitch=drowsiness.head_pitch,
                    yaw=drowsiness.head_yaw or 0,
                    roll=drowsiness.head_roll or 0
                )
                distraction = self._head_pose_estimator.detect_distraction(head_pose)
        
        self._update_stats(drowsiness, distraction, dt)
        self._check_and_emit_events(drowsiness, distraction)
        is_safe, safety_score, alerts = self._calculate_safety(drowsiness, distraction)
        
        state = DriverSafetyState(
            drowsiness=drowsiness,
            distraction=distraction,
            is_safe=is_safe,
            safety_score=safety_score,
            active_alerts=alerts,
            session_start=self._session_start,
            total_drowsy_seconds=self._total_drowsy_seconds,
            total_distracted_seconds=self._total_distracted_seconds,
            drowsiness_events=self._drowsiness_events,
            distraction_events=self._distraction_events
        )
        
        self._current_state = state
        return state
    
    def _update_stats(
        self,
        drowsiness: DrowsinessState,
        distraction: Optional[DistractionState],
        dt: float
    ):
        """Update session statistics"""
        # Track drowsy time
        if drowsiness.is_drowsy:
            self._total_drowsy_seconds += dt
            if not self._last_drowsy_state:
                self._drowsiness_events += 1
        self._last_drowsy_state = drowsiness.is_drowsy
        
        # Track distracted time
        if distraction and distraction.is_distracted:
            self._total_distracted_seconds += dt
            if not self._last_distracted_state:
                self._distraction_events += 1
        self._last_distracted_state = distraction.is_distracted if distraction else False
    
    def _check_and_emit_events(
        self,
        drowsiness: DrowsinessState,
        distraction: Optional[DistractionState]
    ):
        """Check for alert conditions and emit events"""
        now = datetime.now(timezone.utc)
        
        # Drowsiness events
        if drowsiness.alert_level == AlertLevel.WARNING and not self._last_drowsy_state:
            self._emit_event(SafetyEvent(
                event_type=SafetyEventType.DROWSINESS_WARNING,
                severity="warning",
                timestamp=now,
                duration_seconds=drowsiness.closed_duration_ms / 1000,
                details={
                    "ear_average": drowsiness.ear_average,
                    "blinks_per_minute": drowsiness.blinks_per_minute
                }
            ))
        
        if drowsiness.alert_level == AlertLevel.ALERT:
            self._emit_event(SafetyEvent(
                event_type=SafetyEventType.DROWSINESS_ALERT,
                severity="warning",
                timestamp=now,
                duration_seconds=drowsiness.closed_duration_ms / 1000,
                details={"ear_average": drowsiness.ear_average}
            ))
        
        if drowsiness.alert_level == AlertLevel.CRITICAL:
            self._emit_event(SafetyEvent(
                event_type=SafetyEventType.DROWSINESS_CRITICAL,
                severity="critical",
                timestamp=now,
                duration_seconds=drowsiness.closed_duration_ms / 1000,
                details={"ear_average": drowsiness.ear_average}
            ))
        
        # High blink rate
        if drowsiness.blinks_per_minute > 25:
            self._emit_event(SafetyEvent(
                event_type=SafetyEventType.HIGH_BLINK_RATE,
                severity="info",
                timestamp=now,
                duration_seconds=0,
                details={"blinks_per_minute": drowsiness.blinks_per_minute}
            ))
        
        # Distraction events
        if distraction and distraction.is_distracted and not self._last_distracted_state:
            self._emit_event(SafetyEvent(
                event_type=SafetyEventType.DISTRACTION_ALERT,
                severity="warning",
                timestamp=now,
                duration_seconds=distraction.distraction_duration_ms / 1000,
                details={
                    "distraction_type": distraction.distraction_type.value,
                    "head_pose": distraction.head_pose.to_dict()
                }
            ))
        
        # Face not detected
        if not drowsiness.face_detected:
            self._emit_event(SafetyEvent(
                event_type=SafetyEventType.FACE_NOT_DETECTED,
                severity="info",
                timestamp=now,
                duration_seconds=0,
                details={}
            ))
    
    def _emit_event(self, event: SafetyEvent):
        """Emit a safety event to all registered callbacks"""
        self._events.append(event)
        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _calculate_safety(
        self,
        drowsiness: DrowsinessState,
        distraction: Optional[DistractionState]
    ) -> tuple:
        """Calculate overall safety score and alerts"""
        safety_score = 100.0
        alerts = []
        
        # Drowsiness impact
        if drowsiness.alert_level == AlertLevel.CRITICAL:
            safety_score -= 50
            alerts.append("CRITICAL: Eyes closed too long!")
        elif drowsiness.alert_level == AlertLevel.ALERT:
            safety_score -= 30
            alerts.append("ALERT: Drowsiness detected")
        elif drowsiness.alert_level == AlertLevel.WARNING:
            safety_score -= 15
            alerts.append("WARNING: Eyes closing")
        
        # Blink rate impact
        if drowsiness.blinks_per_minute > 25:
            safety_score -= 10
            alerts.append("High blink rate - fatigue indicator")
        
        # Distraction impact
        if distraction:
            if distraction.is_distracted:
                safety_score -= 25
                alerts.append(f"Distracted: {distraction.distraction_type.value}")
            elif not distraction.looking_at_road:
                safety_score -= 10
        
        # Face detection
        if not drowsiness.face_detected:
            safety_score -= 20
            alerts.append("Face not detected")
        
        safety_score = max(0, safety_score)
        is_safe = safety_score >= 70 and drowsiness.alert_level not in [AlertLevel.ALERT, AlertLevel.CRITICAL]
        
        return is_safe, safety_score, alerts
    
    def _disabled_state(self) -> DriverSafetyState:
        """Return a default state when detection is disabled"""
        return DriverSafetyState(
            drowsiness=DrowsinessState(
                ear_left=0, ear_right=0, ear_average=0,
                eyes_closed=False, closed_duration_ms=0,
                blink_count=0, blinks_per_minute=0,
                alert_level=AlertLevel.NONE, is_drowsy=False,
                face_detected=False, confidence=0, timestamp=time.time()
            ),
            is_safe=True,
            safety_score=100,
            active_alerts=["Drowsiness detection disabled"]
        )
    
    def get_state(self) -> Optional[DriverSafetyState]:
        """Get the current state"""
        return self._current_state
    
    def get_events(self) -> List[SafetyEvent]:
        """Get all events from current session"""
        return self._events.copy()
    
    def get_session_summary(self) -> Dict:
        """Get a summary of the current monitoring session"""
        session_duration = time.time() - self._session_start
        
        return {
            "session_duration_seconds": round(session_duration, 1),
            "total_drowsy_seconds": round(self._total_drowsy_seconds, 1),
            "total_distracted_seconds": round(self._total_distracted_seconds, 1),
            "drowsy_percentage": round((self._total_drowsy_seconds / session_duration) * 100, 1) if session_duration > 0 else 0,
            "distracted_percentage": round((self._total_distracted_seconds / session_duration) * 100, 1) if session_duration > 0 else 0,
            "drowsiness_events": self._drowsiness_events,
            "distraction_events": self._distraction_events,
            "total_events": len(self._events),
            "average_safety_score": self._current_state.safety_score if self._current_state else 100
        }
    
    def on_alert(self, callback: Callable[[SafetyEvent], None]):
        """Register a callback for safety alerts"""
        self._alert_callbacks.append(callback)
    
    def reset(self):
        """Reset all tracking state"""
        self._eye_tracker.reset()
        if self._head_pose_estimator:
            self._head_pose_estimator.reset()
        
        self._session_start = time.time()
        self._total_drowsy_seconds = 0.0
        self._total_distracted_seconds = 0.0
        self._drowsiness_events = 0
        self._distraction_events = 0
        self._last_drowsy_state = False
        self._last_distracted_state = False
        self._last_update = time.time()
        self._current_state = None
        self._events = []
        
        logger.info("DrowsinessDetector reset")


# Global instance
_detector: Optional[DrowsinessDetector] = None


def get_drowsiness_detector() -> DrowsinessDetector:
    """Get or create the singleton DrowsinessDetector"""
    global _detector
    if _detector is None:
        _detector = DrowsinessDetector()
    return _detector
