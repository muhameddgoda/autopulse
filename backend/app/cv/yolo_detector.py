"""
AutoPulse YOLO Drowsiness Detector
==================================
Primary drowsiness detection using YOLOv8/YOLOv5 with MediaPipe fallback.

This module provides robust drowsiness detection by:
1. Using YOLO for primary face/eye state detection (more robust)
2. Falling back to MediaPipe EAR-based detection if YOLO fails
3. Combining both approaches for maximum accuracy

Classes detected by YOLO:
- awake: Driver is alert with eyes open
- drowsy: Driver shows signs of drowsiness (eyes closed/heavy)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import time
import logging
import base64
import os

logger = logging.getLogger(__name__)

# Try to import YOLO (ultralytics for YOLOv8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_VERSION = "v8"
    logger.info("YOLOv8 (ultralytics) loaded successfully")
except ImportError:
    try:
        import torch
        YOLO_AVAILABLE = True
        YOLO_VERSION = "v5"
        logger.info("YOLOv5 (torch.hub) available")
    except ImportError:
        YOLO_AVAILABLE = False
        YOLO_VERSION = None
        logger.warning("YOLO not available. Install with: pip install ultralytics")

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")

# Import MediaPipe fallback
try:
    from app.cv.eye_tracker import EyeTracker, DrowsinessState, AlertLevel, MEDIAPIPE_AVAILABLE
    FALLBACK_AVAILABLE = MEDIAPIPE_AVAILABLE
except ImportError:
    FALLBACK_AVAILABLE = False
    EyeTracker = None
    DrowsinessState = None
    AlertLevel = None


class YOLOAlertLevel(Enum):
    """Alert levels based on YOLO detection"""
    NONE = "none"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class YOLODetection:
    """Single YOLO detection result"""
    class_name: str          # "awake" or "drowsy"
    confidence: float        # Detection confidence 0-1
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    def to_dict(self) -> Dict:
        return {
            "class_name": self.class_name,
            "confidence": float(round(self.confidence, 3)),
            "bbox": [int(x) for x in self.bbox]
        }


@dataclass
class YOLODrowsinessState:
    """Drowsiness state from YOLO detection"""
    # Detection results
    is_drowsy: bool
    drowsy_confidence: float
    awake_confidence: float
    
    # Alert status
    alert_level: YOLOAlertLevel
    
    # Timing
    drowsy_duration_ms: float
    
    # Detection metadata
    face_detected: bool
    detection_method: str  # "yolo" or "mediapipe"
    detections: List[YOLODetection] = field(default_factory=list)
    
    # Landmarks for visualization (from YOLO bbox or MediaPipe)
    landmarks: Optional[Dict] = None
    
    # Blink tracking (from MediaPipe fallback)
    blink_count: int = 0
    blinks_per_minute: float = 0.0
    
    # Raw EAR values (from MediaPipe fallback, None if using YOLO)
    ear_average: Optional[float] = None
    
    # Timestamp
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        result = {
            "is_drowsy": bool(self.is_drowsy),
            "drowsy_confidence": float(round(self.drowsy_confidence, 3)),
            "awake_confidence": float(round(self.awake_confidence, 3)),
            "alert_level": self.alert_level.value,
            "drowsy_duration_ms": float(round(self.drowsy_duration_ms, 1)),
            "face_detected": bool(self.face_detected),
            "detection_method": self.detection_method,
            "detections": [d.to_dict() for d in self.detections],
            "blink_count": int(self.blink_count),
            "blinks_per_minute": float(round(self.blinks_per_minute, 1)),
            "ear_average": float(round(self.ear_average, 3)) if self.ear_average is not None else None,
            "timestamp": float(self.timestamp),
            # Compatibility with existing frontend
            "eyes_closed": bool(self.is_drowsy),
            "closed_duration_ms": float(round(self.drowsy_duration_ms, 1)),
            "confidence": float(round(max(self.drowsy_confidence, self.awake_confidence), 3)),
        }
        if self.landmarks:
            result["landmarks"] = self.landmarks
        return result


class YOLODrowsinessDetector:
    """
    YOLO-based drowsiness detection with MediaPipe fallback.
    
    This detector uses YOLO as the primary detection method for robustness,
    with MediaPipe as a fallback for additional eye tracking features.
    
    Usage:
        detector = YOLODrowsinessDetector()
        
        # Process frame
        state = detector.process_frame(frame)
        
        # Or process base64
        state = detector.process_base64(base64_data)
    """
    
    # Alert thresholds
    DROWSY_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider drowsy
    WARNING_DURATION_MS = 500          # 0.5 seconds
    ALERT_DURATION_MS = 2000           # 2 seconds
    CRITICAL_DURATION_MS = 4000        # 4 seconds
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_fallback: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the YOLO drowsiness detector.
        
        Args:
            model_path: Path to custom YOLO weights. If None, uses default/downloads.
            use_fallback: Whether to use MediaPipe as fallback
            confidence_threshold: Minimum detection confidence
        """
        self._enabled = False
        self._model = None
        self._fallback_tracker: Optional[EyeTracker] = None
        self._confidence_threshold = confidence_threshold
        
        # State tracking
        self._drowsy_start: Optional[float] = None
        self._last_state: Optional[YOLODrowsinessState] = None
        self._session_start = time.time()
        self._total_drowsy_seconds = 0.0
        self._drowsiness_events = 0
        self._last_drowsy = False
        self._last_update = time.time()
        
        # Initialize YOLO
        if YOLO_AVAILABLE and CV2_AVAILABLE:
            try:
                self._init_yolo(model_path)
                self._enabled = True
                logger.info(f"YOLO detector initialized (version: {YOLO_VERSION})")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO: {e}")
        
        # Initialize fallback
        if use_fallback and FALLBACK_AVAILABLE:
            try:
                self._fallback_tracker = EyeTracker()
                logger.info("MediaPipe fallback initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe fallback: {e}")
        
        if not self._enabled and not self._fallback_tracker:
            logger.error("No detection method available!")
    
    def _init_yolo(self, model_path: Optional[str] = None):
        """Initialize YOLO model"""
        if YOLO_VERSION == "v8":
            # YOLOv8 with ultralytics
            if model_path and os.path.exists(model_path):
                self._model = YOLO(model_path)
                logger.info(f"Loaded custom YOLOv8 model from {model_path}")
            else:
                # Use YOLOv8n (nano) for face detection as base
                # In production, you'd train a custom model with awake/drowsy classes
                self._model = YOLO('yolov8n.pt')
                logger.info("Loaded default YOLOv8n model")
                
        elif YOLO_VERSION == "v5":
            # YOLOv5 with torch.hub
            import torch
            if model_path and os.path.exists(model_path):
                self._model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                            path=model_path, force_reload=False)
                logger.info(f"Loaded custom YOLOv5 model from {model_path}")
            else:
                self._model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                logger.info("Loaded default YOLOv5s model")
    
    @property
    def enabled(self) -> bool:
        return self._enabled or self._fallback_tracker is not None
    
    @property
    def detection_method(self) -> str:
        if self._enabled:
            return f"yolo_{YOLO_VERSION}"
        elif self._fallback_tracker:
            return "mediapipe"
        return "none"
    
    def process_frame(self, frame: np.ndarray) -> YOLODrowsinessState:
        """
        Process a video frame and return drowsiness state.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            YOLODrowsinessState with detection results
        """
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        # Try YOLO first
        if self._enabled and self._model is not None:
            state = self._process_with_yolo(frame, current_time)
            if state.face_detected:
                self._update_stats(state, dt)
                self._last_state = state
                return state
        
        # Fallback to MediaPipe
        if self._fallback_tracker:
            state = self._process_with_mediapipe(frame, current_time)
            self._update_stats(state, dt)
            self._last_state = state
            return state
        
        # No detection available
        return self._empty_state(current_time)
    
    def _process_with_yolo(self, frame: np.ndarray, current_time: float) -> YOLODrowsinessState:
        """Process frame with YOLO model"""
        try:
            # Run inference
            if YOLO_VERSION == "v8":
                results = self._model(frame, verbose=False)
                detections = self._parse_yolov8_results(results, frame.shape)
            else:
                results = self._model(frame)
                detections = self._parse_yolov5_results(results, frame.shape)
            
            # Analyze detections
            is_drowsy = False
            drowsy_conf = 0.0
            awake_conf = 0.0
            face_detected = len(detections) > 0
            landmarks = None
            
            for det in detections:
                if det.class_name == "drowsy":
                    drowsy_conf = max(drowsy_conf, det.confidence)
                    is_drowsy = det.confidence > self._confidence_threshold
                elif det.class_name == "awake":
                    awake_conf = max(awake_conf, det.confidence)
                elif det.class_name == "person":
                    # Generic person detection - use as face proxy
                    face_detected = True
                    # Create bbox-based landmarks for visualization
                    x1, y1, x2, y2 = det.bbox
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    w, h = x2 - x1, y2 - y1
                    # Estimate eye positions (roughly 1/3 down from top, 1/4 from sides)
                    left_eye_x = x1 + w // 4
                    right_eye_x = x2 - w // 4
                    eye_y = y1 + h // 3
                    landmarks = {
                        "left_eye": [[left_eye_x - 10, eye_y], [left_eye_x, eye_y - 5], 
                                    [left_eye_x + 10, eye_y], [left_eye_x, eye_y + 5]],
                        "right_eye": [[right_eye_x - 10, eye_y], [right_eye_x, eye_y - 5],
                                     [right_eye_x + 10, eye_y], [right_eye_x, eye_y + 5]]
                    }
            
            # Calculate drowsy duration
            if is_drowsy:
                if self._drowsy_start is None:
                    self._drowsy_start = current_time
                drowsy_duration_ms = (current_time - self._drowsy_start) * 1000
            else:
                drowsy_duration_ms = 0
                self._drowsy_start = None
            
            # Determine alert level
            alert_level = self._calculate_alert_level(is_drowsy, drowsy_duration_ms)
            
            return YOLODrowsinessState(
                is_drowsy=is_drowsy,
                drowsy_confidence=drowsy_conf,
                awake_confidence=awake_conf,
                alert_level=alert_level,
                drowsy_duration_ms=drowsy_duration_ms,
                face_detected=face_detected,
                detection_method="yolo",
                detections=detections,
                landmarks=landmarks,
                timestamp=current_time
            )
            
        except Exception as e:
            logger.error(f"YOLO processing error: {e}")
            return self._empty_state(current_time, face_detected=False)
    
    def _parse_yolov8_results(self, results, frame_shape) -> List[YOLODetection]:
        """Parse YOLOv8 results"""
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Get class name
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append(YOLODetection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    ))
        return detections
    
    def _parse_yolov5_results(self, results, frame_shape) -> List[YOLODetection]:
        """Parse YOLOv5 results"""
        detections = []
        preds = results.xyxy[0].cpu().numpy()
        names = results.names
        
        for pred in preds:
            x1, y1, x2, y2, conf, cls_id = pred
            class_name = names.get(int(cls_id), f"class_{int(cls_id)}")
            
            detections.append(YOLODetection(
                class_name=class_name,
                confidence=float(conf),
                bbox=(int(x1), int(y1), int(x2), int(y2))
            ))
        return detections
    
    def _process_with_mediapipe(self, frame: np.ndarray, current_time: float) -> YOLODrowsinessState:
        """Process frame with MediaPipe fallback"""
        mp_state = self._fallback_tracker.process_frame(frame)
        
        # Convert MediaPipe state to YOLO state
        is_drowsy = mp_state.is_drowsy
        
        # Calculate drowsy duration
        if is_drowsy:
            if self._drowsy_start is None:
                self._drowsy_start = current_time
            drowsy_duration_ms = (current_time - self._drowsy_start) * 1000
        else:
            drowsy_duration_ms = mp_state.closed_duration_ms
            if not mp_state.eyes_closed:
                self._drowsy_start = None
        
        alert_level = self._calculate_alert_level(is_drowsy, drowsy_duration_ms)
        
        # Get landmarks from MediaPipe
        landmarks = None
        if hasattr(mp_state, 'landmarks') and mp_state.landmarks:
            landmarks = mp_state.landmarks
        
        return YOLODrowsinessState(
            is_drowsy=is_drowsy,
            drowsy_confidence=1.0 if is_drowsy else 0.0,
            awake_confidence=0.0 if is_drowsy else 1.0,
            alert_level=alert_level,
            drowsy_duration_ms=drowsy_duration_ms,
            face_detected=mp_state.face_detected,
            detection_method="mediapipe",
            detections=[],
            landmarks=landmarks,
            blink_count=mp_state.blink_count,
            blinks_per_minute=mp_state.blinks_per_minute,
            ear_average=mp_state.ear_average,
            timestamp=current_time
        )
    
    def _calculate_alert_level(self, is_drowsy: bool, duration_ms: float) -> YOLOAlertLevel:
        """Calculate alert level based on drowsiness duration"""
        if not is_drowsy:
            return YOLOAlertLevel.NONE
        
        if duration_ms >= self.CRITICAL_DURATION_MS:
            return YOLOAlertLevel.CRITICAL
        elif duration_ms >= self.ALERT_DURATION_MS:
            return YOLOAlertLevel.ALERT
        elif duration_ms >= self.WARNING_DURATION_MS:
            return YOLOAlertLevel.WARNING
        
        return YOLOAlertLevel.NONE
    
    def _update_stats(self, state: YOLODrowsinessState, dt: float):
        """Update session statistics"""
        if state.is_drowsy:
            self._total_drowsy_seconds += dt
            if not self._last_drowsy:
                self._drowsiness_events += 1
        self._last_drowsy = state.is_drowsy
    
    def _empty_state(self, current_time: float, face_detected: bool = False) -> YOLODrowsinessState:
        """Return empty state when no detection possible"""
        return YOLODrowsinessState(
            is_drowsy=False,
            drowsy_confidence=0.0,
            awake_confidence=0.0,
            alert_level=YOLOAlertLevel.NONE,
            drowsy_duration_ms=0.0,
            face_detected=face_detected,
            detection_method="none",
            detections=[],
            timestamp=current_time
        )
    
    def process_base64(self, base64_data: str) -> YOLODrowsinessState:
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
    
    def get_session_summary(self) -> Dict:
        """Get session statistics"""
        duration = time.time() - self._session_start
        return {
            "session_duration_seconds": round(duration, 1),
            "total_drowsy_seconds": round(self._total_drowsy_seconds, 1),
            "drowsy_percentage": round((self._total_drowsy_seconds / duration) * 100, 1) if duration > 0 else 0,
            "drowsiness_events": self._drowsiness_events,
            "detection_method": self.detection_method,
            "average_safety_score": 100 - (self._total_drowsy_seconds / duration * 50) if duration > 0 else 100
        }
    
    def reset(self):
        """Reset all tracking state"""
        self._drowsy_start = None
        self._last_state = None
        self._session_start = time.time()
        self._total_drowsy_seconds = 0.0
        self._drowsiness_events = 0
        self._last_drowsy = False
        self._last_update = time.time()
        
        if self._fallback_tracker:
            self._fallback_tracker.reset()
        
        logger.info("YOLODrowsinessDetector reset")


# Singleton instance
_yolo_detector: Optional[YOLODrowsinessDetector] = None


def get_yolo_detector(model_path: Optional[str] = None) -> YOLODrowsinessDetector:
    """Get or create the singleton YOLODrowsinessDetector"""
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YOLODrowsinessDetector(model_path=model_path)
    return _yolo_detector


# Check availability
def is_yolo_available() -> bool:
    """Check if YOLO is available"""
    return YOLO_AVAILABLE and CV2_AVAILABLE
