"""
AutoPulse Computer Vision Module
================================
Driver safety monitoring using computer vision.

Components:
- EyeTracker: Drowsiness detection using Eye Aspect Ratio (EAR) with MediaPipe
- HeadPoseEstimator: Distraction detection using head pose
- YOLODrowsinessDetector: YOLO-based drowsiness detection (primary)
- DrowsinessDetector: High-level service combining all CV features
- UnifiedDetector: Smart detector that uses YOLO primary + MediaPipe fallback
"""

# MediaPipe-based components
from app.cv.eye_tracker import (
    EyeTracker,
    DrowsinessState,
    AlertLevel,
    EyeMetrics,
    MEDIAPIPE_AVAILABLE,
    get_eye_tracker
)

from app.cv.head_pose import (
    HeadPoseEstimator,
    HeadPose,
    DistractionState,
    DistractionType
)

from app.cv.drowsiness_detector import (
    DrowsinessDetector,
    DriverSafetyState,
    SafetyEvent,
    SafetyEventType,
    get_drowsiness_detector
)

# YOLO-based components
from app.cv.yolo_detector import (
    YOLODrowsinessDetector,
    YOLODrowsinessState,
    YOLODetection,
    YOLOAlertLevel,
    get_yolo_detector,
    is_yolo_available,
    YOLO_AVAILABLE
)

from app.cv.eye_detector import (
    EyeDrowsinessDetector,
    EyeDetectionState,
    get_eye_detector,
    is_eye_detector_available
)

# Unified detector (smart selection)
from app.cv.unified_detector import (
    UnifiedDrowsinessDetector,
    get_unified_detector,
    DetectionMethod
)

__all__ = [
    # Eye tracking (MediaPipe)
    "EyeTracker",
    "DrowsinessState", 
    "AlertLevel",
    "EyeMetrics",
    "MEDIAPIPE_AVAILABLE",
    "get_eye_tracker",
    
    # Head pose
    "HeadPoseEstimator",
    "HeadPose",
    "DistractionState",
    "DistractionType",
    
    # MediaPipe detector
    "DrowsinessDetector",
    "DriverSafetyState",
    "SafetyEvent",
    "SafetyEventType",
    "get_drowsiness_detector",
    
    # YOLO detector
    "YOLODrowsinessDetector",
    "YOLODrowsinessState",
    "YOLODetection",
    "YOLOAlertLevel",
    "get_yolo_detector",
    "is_yolo_available",
    "YOLO_AVAILABLE",
    
    # Unified detector
    "UnifiedDrowsinessDetector",
    "get_unified_detector",
    "DetectionMethod",

    "EyeDrowsinessDetector",
    "EyeDetectionState", 
    "get_eye_detector",
    "is_eye_detector_available",
]

# Add this import
