"""
AutoPulse Unified Drowsiness Detector
=====================================
Smart detector that combines YOLO and MediaPipe for maximum robustness.

Strategy:
1. Try YOLO first (more robust to eye closure)
2. If YOLO fails or no custom model, use MediaPipe
3. Combine results for enhanced accuracy
4. Provide consistent API regardless of backend
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import time
import logging
import base64

logger = logging.getLogger(__name__)

# Try imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import detection backends
try:
    from app.cv.yolo_detector import (
        YOLODrowsinessDetector, 
        YOLODrowsinessState,
        is_yolo_available,
        YOLO_AVAILABLE
    )
except ImportError:
    YOLO_AVAILABLE = False
    YOLODrowsinessDetector = None
    is_yolo_available = lambda: False

try:
    from app.cv.eye_tracker import (
        EyeTracker,
        DrowsinessState,
        AlertLevel,
        MEDIAPIPE_AVAILABLE
    )
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    EyeTracker = None


class DetectionMethod(Enum):
    """Available detection methods"""
    YOLO = "yolo"
    MEDIAPIPE = "mediapipe"
    HYBRID = "hybrid"  # YOLO + MediaPipe combined
    AUTO = "auto"      # Automatically select best available


class UnifiedAlertLevel(Enum):
    """Unified alert levels"""
    NONE = "none"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class UnifiedDrowsinessState:
    """
    Unified drowsiness state that works with any detection backend.
    
    This provides a consistent interface regardless of whether YOLO,
    MediaPipe, or both are being used.
    """
    # Core detection
    is_drowsy: bool
    alert_level: UnifiedAlertLevel
    face_detected: bool
    
    # Confidence scores
    drowsy_confidence: float
    detection_confidence: float
    
    # Timing
    drowsy_duration_ms: float
    
    # Eye metrics (from MediaPipe or estimated)
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_average: float = 0.0
    eyes_closed: bool = False
    closed_duration_ms: float = 0.0
    blink_count: int = 0
    blinks_per_minute: float = 0.0
    
    # Head pose (if available)
    head_pitch: Optional[float] = None
    head_yaw: Optional[float] = None
    head_roll: Optional[float] = None
    
    # Detection metadata
    detection_method: str = "auto"
    yolo_detections: List[Dict] = field(default_factory=list)
    
    # Visualization
    landmarks: Optional[Dict] = None
    bounding_boxes: List[Dict] = field(default_factory=list)
    
    # Timestamp
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            # Core (frontend expects these)
            "is_drowsy": bool(self.is_drowsy),
            "alert_level": self.alert_level.value,
            "face_detected": bool(self.face_detected),
            
            # Confidence
            "drowsy_confidence": float(round(self.drowsy_confidence, 3)),
            "confidence": float(round(self.detection_confidence, 3)),
            
            # Eye metrics (frontend compatibility)
            "ear_left": float(round(self.ear_left, 3)),
            "ear_right": float(round(self.ear_right, 3)),
            "ear_average": float(round(self.ear_average, 3)),
            "eyes_closed": bool(self.eyes_closed),
            "closed_duration_ms": float(round(self.closed_duration_ms, 1)),
            "blink_count": int(self.blink_count),
            "blinks_per_minute": float(round(self.blinks_per_minute, 1)),
            
            # Head pose
            "head_pitch": float(round(self.head_pitch, 1)) if self.head_pitch is not None else None,
            "head_yaw": float(round(self.head_yaw, 1)) if self.head_yaw is not None else None,
            "head_roll": float(round(self.head_roll, 1)) if self.head_roll is not None else None,
            
            # Metadata
            "detection_method": self.detection_method,
            "timestamp": float(self.timestamp),
            
            # Visualization
            "landmarks": self.landmarks,
            "bounding_boxes": self.bounding_boxes,
        }


class UnifiedDrowsinessDetector:
    """
    Unified drowsiness detector that intelligently combines detection methods.
    
    This detector provides:
    - Automatic method selection based on availability
    - YOLO as primary (more robust when eyes closed)
    - MediaPipe as fallback/enhancement (better eye metrics)
    - Consistent API regardless of backend
    - Combined confidence scoring
    
    Usage:
        detector = UnifiedDrowsinessDetector(method=DetectionMethod.AUTO)
        state = detector.process_frame(frame)
        # or
        state = detector.process_base64(base64_data)
    """
    
    # Thresholds
    WARNING_DURATION_MS = 500
    ALERT_DURATION_MS = 2000
    CRITICAL_DURATION_MS = 4000
    DROWSY_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(
        self,
        method: DetectionMethod = DetectionMethod.AUTO,
        yolo_model_path: Optional[str] = None,
        use_hybrid: bool = True
    ):
        """
        Initialize the unified detector.
        
        Args:
            method: Detection method to use
            yolo_model_path: Path to custom YOLO model weights
            use_hybrid: If True and both available, combine YOLO + MediaPipe
        """
        self._method = method
        self._use_hybrid = use_hybrid
        
        # Initialize backends
        self._yolo_detector: Optional[YOLODrowsinessDetector] = None
        self._mediapipe_tracker: Optional[EyeTracker] = None
        
        # Try YOLO
        if YOLO_AVAILABLE and method in [DetectionMethod.YOLO, DetectionMethod.HYBRID, DetectionMethod.AUTO]:
            try:
                self._yolo_detector = YOLODrowsinessDetector(
                    model_path=yolo_model_path,
                    use_fallback=False  # We handle fallback ourselves
                )
                logger.info("YOLO detector initialized in unified detector")
            except Exception as e:
                logger.warning(f"Failed to init YOLO: {e}")
        
        # Try MediaPipe
        if MEDIAPIPE_AVAILABLE and method in [DetectionMethod.MEDIAPIPE, DetectionMethod.HYBRID, DetectionMethod.AUTO]:
            try:
                self._mediapipe_tracker = EyeTracker()
                logger.info("MediaPipe tracker initialized in unified detector")
            except Exception as e:
                logger.warning(f"Failed to init MediaPipe: {e}")
        
        # Determine actual method
        if self._yolo_detector and self._mediapipe_tracker and use_hybrid:
            self._active_method = DetectionMethod.HYBRID
        elif self._yolo_detector:
            self._active_method = DetectionMethod.YOLO
        elif self._mediapipe_tracker:
            self._active_method = DetectionMethod.MEDIAPIPE
        else:
            self._active_method = None
            logger.error("No detection method available!")
        
        # State tracking
        self._drowsy_start: Optional[float] = None
        self._session_start = time.time()
        self._total_drowsy_seconds = 0.0
        self._total_distracted_seconds = 0.0
        self._drowsiness_events = 0
        self._distraction_events = 0
        self._last_drowsy = False
        self._last_update = time.time()
        self._current_state: Optional[UnifiedDrowsinessState] = None
        
        logger.info(f"Unified detector ready. Method: {self._active_method}")
    
    @property
    def enabled(self) -> bool:
        return self._active_method is not None
    
    @property
    def detection_method(self) -> str:
        return self._active_method.value if self._active_method else "none"
    
    def process_frame(self, frame: np.ndarray) -> UnifiedDrowsinessState:
        """
        Process a video frame with the best available method.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            UnifiedDrowsinessState with detection results
        """
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        if self._active_method == DetectionMethod.HYBRID:
            state = self._process_hybrid(frame, current_time)
        elif self._active_method == DetectionMethod.YOLO:
            state = self._process_yolo_only(frame, current_time)
        elif self._active_method == DetectionMethod.MEDIAPIPE:
            state = self._process_mediapipe_only(frame, current_time)
        else:
            state = self._empty_state(current_time)
        
        # Update stats
        self._update_stats(state, dt)
        self._current_state = state
        
        return state
    
    def _process_hybrid(self, frame: np.ndarray, current_time: float) -> UnifiedDrowsinessState:
        """Process with both YOLO and MediaPipe, combining results"""
        yolo_state = None
        mp_state = None
        
        # Get YOLO result
        if self._yolo_detector:
            try:
                yolo_state = self._yolo_detector.process_frame(frame)
            except Exception as e:
                logger.debug(f"YOLO processing failed: {e}")
        
        # Get MediaPipe result
        if self._mediapipe_tracker:
            try:
                mp_state = self._mediapipe_tracker.process_frame(frame)
            except Exception as e:
                logger.debug(f"MediaPipe processing failed: {e}")
        
        # Combine results
        return self._combine_results(yolo_state, mp_state, current_time)
    
    def _combine_results(
        self, 
        yolo_state: Optional[Any], 
        mp_state: Optional[DrowsinessState],
        current_time: float
    ) -> UnifiedDrowsinessState:
        """Intelligently combine YOLO and MediaPipe results"""
        
        # Determine drowsiness from both sources
        yolo_drowsy = yolo_state.is_drowsy if yolo_state else False
        yolo_conf = yolo_state.drowsy_confidence if yolo_state else 0.0
        yolo_face = yolo_state.face_detected if yolo_state else False
        
        mp_drowsy = mp_state.is_drowsy if mp_state else False
        mp_face = mp_state.face_detected if mp_state else False
        mp_ear = mp_state.ear_average if mp_state else 0.3
        mp_eyes_closed = mp_state.eyes_closed if mp_state else False
        
        # Combined logic:
        # - If YOLO says drowsy with high confidence, trust it
        # - If MediaPipe says eyes closed, trust it
        # - Combine for final decision
        is_drowsy = False
        combined_conf = 0.0
        
        if yolo_conf > 0.7:
            # High confidence YOLO detection
            is_drowsy = yolo_drowsy
            combined_conf = yolo_conf
        elif mp_state and mp_eyes_closed:
            # MediaPipe detected closed eyes
            is_drowsy = mp_drowsy or mp_state.closed_duration_ms > 500
            combined_conf = 0.8 if mp_state.closed_duration_ms > 500 else 0.5
        elif yolo_drowsy and mp_eyes_closed:
            # Both agree
            is_drowsy = True
            combined_conf = max(yolo_conf, 0.7)
        elif yolo_drowsy or mp_drowsy:
            # One says drowsy
            is_drowsy = yolo_drowsy or mp_drowsy
            combined_conf = max(yolo_conf, 0.5 if mp_drowsy else 0.0)
        
        # Face detection - trust either
        face_detected = yolo_face or mp_face
        
        # Calculate duration
        if is_drowsy:
            if self._drowsy_start is None:
                self._drowsy_start = current_time
            drowsy_duration_ms = (current_time - self._drowsy_start) * 1000
        else:
            drowsy_duration_ms = 0
            self._drowsy_start = None
        
        # Get eye metrics from MediaPipe (more accurate)
        ear_left = mp_state.ear_left if mp_state else 0.0
        ear_right = mp_state.ear_right if mp_state else 0.0
        ear_average = mp_state.ear_average if mp_state else 0.0
        eyes_closed = mp_state.eyes_closed if mp_state else is_drowsy
        closed_duration_ms = mp_state.closed_duration_ms if mp_state else drowsy_duration_ms
        blink_count = mp_state.blink_count if mp_state else 0
        blinks_per_minute = mp_state.blinks_per_minute if mp_state else 0.0
        
        # Head pose from MediaPipe
        head_pitch = mp_state.head_pitch if mp_state else None
        head_yaw = mp_state.head_yaw if mp_state else None
        head_roll = mp_state.head_roll if mp_state else None
        
        # Landmarks - prefer MediaPipe, fallback to YOLO estimation
        landmarks = None
        if mp_state and hasattr(mp_state, 'landmarks') and mp_state.landmarks:
            landmarks = mp_state.landmarks
        elif yolo_state and yolo_state.landmarks:
            landmarks = yolo_state.landmarks
        
        # YOLO bounding boxes
        bounding_boxes = []
        if yolo_state and yolo_state.detections:
            bounding_boxes = [d.to_dict() for d in yolo_state.detections]
        
        # Alert level
        alert_level = self._calculate_alert_level(is_drowsy, drowsy_duration_ms)
        
        return UnifiedDrowsinessState(
            is_drowsy=is_drowsy,
            alert_level=alert_level,
            face_detected=face_detected,
            drowsy_confidence=combined_conf,
            detection_confidence=max(yolo_conf, 0.9 if mp_face else 0.0),
            drowsy_duration_ms=drowsy_duration_ms,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_average=ear_average,
            eyes_closed=eyes_closed,
            closed_duration_ms=closed_duration_ms,
            blink_count=blink_count,
            blinks_per_minute=blinks_per_minute,
            head_pitch=head_pitch,
            head_yaw=head_yaw,
            head_roll=head_roll,
            detection_method="hybrid",
            yolo_detections=bounding_boxes,
            landmarks=landmarks,
            bounding_boxes=bounding_boxes,
            timestamp=current_time
        )
    
    def _process_yolo_only(self, frame: np.ndarray, current_time: float) -> UnifiedDrowsinessState:
        """Process with YOLO only"""
        try:
            yolo_state = self._yolo_detector.process_frame(frame)
            
            is_drowsy = yolo_state.is_drowsy
            
            if is_drowsy:
                if self._drowsy_start is None:
                    self._drowsy_start = current_time
                drowsy_duration_ms = (current_time - self._drowsy_start) * 1000
            else:
                drowsy_duration_ms = 0
                self._drowsy_start = None
            
            alert_level = self._calculate_alert_level(is_drowsy, drowsy_duration_ms)
            
            return UnifiedDrowsinessState(
                is_drowsy=is_drowsy,
                alert_level=alert_level,
                face_detected=yolo_state.face_detected,
                drowsy_confidence=yolo_state.drowsy_confidence,
                detection_confidence=max(yolo_state.drowsy_confidence, yolo_state.awake_confidence),
                drowsy_duration_ms=drowsy_duration_ms,
                eyes_closed=is_drowsy,
                closed_duration_ms=drowsy_duration_ms,
                detection_method="yolo",
                yolo_detections=[d.to_dict() for d in yolo_state.detections],
                landmarks=yolo_state.landmarks,
                bounding_boxes=[d.to_dict() for d in yolo_state.detections],
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"YOLO processing error: {e}")
            return self._empty_state(current_time)
    
    def _process_mediapipe_only(self, frame: np.ndarray, current_time: float) -> UnifiedDrowsinessState:
        """Process with MediaPipe only"""
        try:
            mp_state = self._mediapipe_tracker.process_frame(frame)
            
            is_drowsy = mp_state.is_drowsy
            
            if is_drowsy:
                if self._drowsy_start is None:
                    self._drowsy_start = current_time
                drowsy_duration_ms = (current_time - self._drowsy_start) * 1000
            else:
                drowsy_duration_ms = mp_state.closed_duration_ms
                if not mp_state.eyes_closed:
                    self._drowsy_start = None
            
            alert_level = self._calculate_alert_level(is_drowsy, drowsy_duration_ms)
            
            landmarks = None
            if hasattr(mp_state, 'landmarks') and mp_state.landmarks:
                landmarks = mp_state.landmarks
            
            return UnifiedDrowsinessState(
                is_drowsy=is_drowsy,
                alert_level=alert_level,
                face_detected=mp_state.face_detected,
                drowsy_confidence=1.0 if is_drowsy else 0.0,
                detection_confidence=mp_state.confidence,
                drowsy_duration_ms=drowsy_duration_ms,
                ear_left=mp_state.ear_left,
                ear_right=mp_state.ear_right,
                ear_average=mp_state.ear_average,
                eyes_closed=mp_state.eyes_closed,
                closed_duration_ms=mp_state.closed_duration_ms,
                blink_count=mp_state.blink_count,
                blinks_per_minute=mp_state.blinks_per_minute,
                head_pitch=mp_state.head_pitch,
                head_yaw=mp_state.head_yaw,
                head_roll=mp_state.head_roll,
                detection_method="mediapipe",
                landmarks=landmarks,
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"MediaPipe processing error: {e}")
            return self._empty_state(current_time)
    
    def _calculate_alert_level(self, is_drowsy: bool, duration_ms: float) -> UnifiedAlertLevel:
        """Calculate alert level based on drowsiness duration"""
        if not is_drowsy:
            return UnifiedAlertLevel.NONE
        
        if duration_ms >= self.CRITICAL_DURATION_MS:
            return UnifiedAlertLevel.CRITICAL
        elif duration_ms >= self.ALERT_DURATION_MS:
            return UnifiedAlertLevel.ALERT
        elif duration_ms >= self.WARNING_DURATION_MS:
            return UnifiedAlertLevel.WARNING
        
        return UnifiedAlertLevel.NONE
    
    def _update_stats(self, state: UnifiedDrowsinessState, dt: float):
        """Update session statistics"""
        if state.is_drowsy:
            self._total_drowsy_seconds += dt
            if not self._last_drowsy:
                self._drowsiness_events += 1
        self._last_drowsy = state.is_drowsy
    
    def _empty_state(self, current_time: float) -> UnifiedDrowsinessState:
        """Return empty state when no detection possible"""
        return UnifiedDrowsinessState(
            is_drowsy=False,
            alert_level=UnifiedAlertLevel.NONE,
            face_detected=False,
            drowsy_confidence=0.0,
            detection_confidence=0.0,
            drowsy_duration_ms=0.0,
            detection_method="none",
            timestamp=current_time
        )
    
    def process_base64(self, base64_data: str) -> UnifiedDrowsinessState:
        """Process a base64-encoded image"""
        if not CV2_AVAILABLE:
            return self._empty_state(time.time())
        
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            # Decode
            img_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return self._empty_state(time.time())
            
            return self.process_frame(frame)
            
        except Exception as e:
            logger.error(f"Base64 processing error: {e}")
            return self._empty_state(time.time())
    
    def get_state(self) -> Optional[UnifiedDrowsinessState]:
        """Get current state"""
        return self._current_state
    
    def get_session_summary(self) -> Dict:
        """Get session statistics"""
        duration = time.time() - self._session_start
        safety_score = 100 - min(50, (self._total_drowsy_seconds / max(duration, 1)) * 100)
        
        return {
            "session_duration_seconds": round(duration, 1),
            "total_drowsy_seconds": round(self._total_drowsy_seconds, 1),
            "total_distracted_seconds": round(self._total_distracted_seconds, 1),
            "drowsy_percentage": round((self._total_drowsy_seconds / max(duration, 1)) * 100, 1),
            "drowsiness_events": self._drowsiness_events,
            "distraction_events": self._distraction_events,
            "detection_method": self.detection_method,
            "average_safety_score": round(safety_score, 1)
        }
    
    def get_events(self) -> List:
        """Get events (compatibility method)"""
        return []  # TODO: implement event tracking
    
    def reset(self):
        """Reset all tracking state"""
        self._drowsy_start = None
        self._session_start = time.time()
        self._total_drowsy_seconds = 0.0
        self._total_distracted_seconds = 0.0
        self._drowsiness_events = 0
        self._distraction_events = 0
        self._last_drowsy = False
        self._last_update = time.time()
        self._current_state = None
        
        if self._mediapipe_tracker:
            self._mediapipe_tracker.reset()
        if self._yolo_detector:
            self._yolo_detector.reset()
        
        logger.info("UnifiedDrowsinessDetector reset")


# Singleton instance
_unified_detector: Optional[UnifiedDrowsinessDetector] = None


def get_unified_detector(
    method: DetectionMethod = DetectionMethod.AUTO,
    yolo_model_path: Optional[str] = None
) -> UnifiedDrowsinessDetector:
    """Get or create the singleton UnifiedDrowsinessDetector"""
    global _unified_detector
    if _unified_detector is None:
        _unified_detector = UnifiedDrowsinessDetector(
            method=method,
            yolo_model_path=yolo_model_path
        )
    return _unified_detector
