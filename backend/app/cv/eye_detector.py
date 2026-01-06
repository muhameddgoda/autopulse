"""
AutoPulse Eye-Based Drowsiness Detector
=======================================
Production-ready detector that integrates with the FastAPI backend.

Combines:
1. MediaPipe for face detection and eye landmark extraction
2. Trained CNN for eye state classification (open/closed)
3. EAR as backup confirmation
4. Temporal smoothing for stable results

This replaces the old full-face approach with accurate eye-only detection.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import logging
import base64

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")


class AlertLevel(Enum):
    """Alert severity levels"""
    NONE = "none"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class EyeDetectionState:
    """Detection state from eye classifier"""
    # Eye state
    eye_state: str  # "open", "closed", "unknown"
    eyes_closed: bool
    confidence: float
    
    # EAR metrics
    ear_left: float
    ear_right: float
    ear_average: float
    
    # Timing
    closed_duration_ms: float
    
    # Blink tracking
    blink_count: int
    blinks_per_minute: float
    
    # Alert status
    alert_level: AlertLevel
    is_drowsy: bool
    
    # Face detection
    face_detected: bool
    
    # Head pose (from MediaPipe)
    head_pitch: Optional[float] = None
    head_yaw: Optional[float] = None
    head_roll: Optional[float] = None
    
    # Visualization data
    landmarks: Optional[Dict] = None
    eye_bboxes: Optional[tuple] = None
    
    # Metadata
    detection_method: str = "eye_classifier"
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "eye_state": self.eye_state,
            "eyes_closed": bool(self.eyes_closed),
            "confidence": float(round(self.confidence, 3)),
            "ear_left": float(round(self.ear_left, 3)),
            "ear_right": float(round(self.ear_right, 3)),
            "ear_average": float(round(self.ear_average, 3)),
            "closed_duration_ms": float(round(self.closed_duration_ms, 1)),
            "blink_count": int(self.blink_count),
            "blinks_per_minute": float(round(self.blinks_per_minute, 1)),
            "alert_level": self.alert_level.value,
            "is_drowsy": bool(self.is_drowsy),
            "face_detected": bool(self.face_detected),
            "head_pitch": float(round(self.head_pitch, 1)) if self.head_pitch else None,
            "head_yaw": float(round(self.head_yaw, 1)) if self.head_yaw else None,
            "head_roll": float(round(self.head_roll, 1)) if self.head_roll else None,
            "detection_method": self.detection_method,
            "timestamp": float(self.timestamp),
            "landmarks": self.landmarks,
        }


# ============================================================================
# CNN Models (must match training architecture)
# ============================================================================

class EyeClassifierCNN(nn.Module):
    """Custom lightweight CNN for eye classification"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


class EyeClassifierMobileNet(nn.Module):
    """MobileNetV2-based classifier"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=False)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# Main Detector Class
# ============================================================================

class EyeDrowsinessDetector:
    """
    Production drowsiness detector using eye classification.
    
    Usage:
        detector = EyeDrowsinessDetector(model_path="models/eye_classifier.pth")
        state = detector.process_frame(frame)
        # or
        state = detector.process_base64(base64_data)
    """
    
    # MediaPipe landmark indices
    LEFT_EYE_EXTENDED = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336, 362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EXTENDED = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 33, 160, 158, 133, 153, 144]
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    
    # Thresholds
    EAR_THRESHOLD = 0.22
    WARNING_DURATION_MS = 500
    ALERT_DURATION_MS = 2000
    CRITICAL_DURATION_MS = 4000
    
    # Image sizes (must match training)
    IMG_HEIGHT = 64
    IMG_WIDTH = 128
    
    def __init__(self, model_path: str = "models/eye_classifier.pth"):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained eye classifier model
        """
        self._enabled = False
        self._model = None
        self._device = None
        self._classes = ["closed", "open"]
        
        # Initialize components
        if TORCH_AVAILABLE and CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
            try:
                self._init_model(model_path)
                self._init_mediapipe()
                self._init_transforms()
                self._enabled = True
                logger.info("EyeDrowsinessDetector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
        else:
            logger.error("Missing dependencies for EyeDrowsinessDetector")
        
        # State tracking
        self._closed_start: Optional[float] = None
        self._closed_duration_ms = 0.0
        self._blink_count = 0
        self._blink_timestamps: List[float] = []
        self._last_eye_state = "open"
        self._prediction_history: List[str] = []
        self._history_size = 5
        
        # Session stats
        self._session_start = time.time()
        self._total_closed_time = 0.0
        self._drowsiness_events = 0
        self._last_drowsy = False
        self._last_update = time.time()
    
    def _init_model(self, model_path: str):
        """Load the trained model"""
        import os
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, using EAR-only mode")
            return
        
        checkpoint = torch.load(model_path, map_location=self._device)
        
        model_type = checkpoint.get("model_type", "custom")
        if model_type == "mobilenet":
            self._model = EyeClassifierMobileNet()
        else:
            self._model = EyeClassifierCNN()
        
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self._device)
        self._model.eval()
        
        self._classes = checkpoint.get("classes", ["closed", "open"])
        
        logger.info(f"Loaded eye classifier from {model_path}")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Classes: {self._classes}")
        logger.info(f"  Device: {self._device}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh"""
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def _init_transforms(self):
        """Initialize image preprocessing"""
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.IMG_HEIGHT, self.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def _get_eye_bbox(self, landmarks, indices, shape) -> tuple:
        """Get bounding box for eye region"""
        h, w = shape[:2]
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        xs, ys = zip(*points)
        
        pad = int((max(xs) - min(xs)) * 0.4)
        return (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(w, max(xs) + pad),
            min(h, max(ys) + pad)
        )
    
    def _calculate_ear(self, landmarks, indices, shape) -> float:
        """Calculate Eye Aspect Ratio"""
        h, w = shape[:2]
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        
        return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0
    
    def _extract_eyes(self, frame, landmarks) -> tuple:
        """Extract combined eye region for classification"""
        left_bbox = self._get_eye_bbox(landmarks, self.LEFT_EYE_EXTENDED, frame.shape)
        right_bbox = self._get_eye_bbox(landmarks, self.RIGHT_EYE_EXTENDED, frame.shape)
        
        left_eye = frame[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
        right_eye = frame[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]
        
        if left_eye.size == 0 or right_eye.size == 0:
            return None, (left_bbox, right_bbox)
        
        # Resize and combine
        target_h = 64
        try:
            left_resized = cv2.resize(left_eye, (int(target_h * left_eye.shape[1] / left_eye.shape[0]), target_h))
            right_resized = cv2.resize(right_eye, (int(target_h * right_eye.shape[1] / right_eye.shape[0]), target_h))
            combined = np.hstack([left_resized, right_resized])
            return combined, (left_bbox, right_bbox)
        except:
            return None, (left_bbox, right_bbox)
    
    def _classify_eyes(self, eye_img) -> Dict:
        """Classify eye state using CNN"""
        if self._model is None:
            return {"class": "unknown", "confidence": 0.0}
        
        rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            output = self._model(tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        pred_idx = probs.argmax().item()
        
        return {
            "class": self._classes[pred_idx],
            "confidence": float(probs[pred_idx].item())
        }
    
    def _get_head_pose(self, landmarks, shape) -> tuple:
        """Estimate head pose from landmarks"""
        h, w = shape[:2]
        
        # Key points for pose estimation
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Simple pose estimation based on landmark positions
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        
        # Yaw (left-right rotation)
        yaw = (nose.x - eye_center_x) * 100
        
        # Pitch (up-down rotation)  
        pitch = (nose.y - eye_center_y) * 100
        
        # Roll (tilt)
        roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x) * 180 / np.pi
        
        return pitch, yaw, roll
    
    def _get_eye_landmarks_for_viz(self, landmarks, shape) -> Dict:
        """Extract eye landmarks for frontend visualization"""
        h, w = shape[:2]
        
        left_eye_pts = [[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                        for i in self.LEFT_EYE_EAR]
        right_eye_pts = [[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                         for i in self.RIGHT_EYE_EAR]
        
        return {
            "left_eye": left_eye_pts,
            "right_eye": right_eye_pts
        }
    
    def _update_blink_stats(self, current_time: float):
        """Update blink count and rate"""
        # Remove old timestamps (older than 60 seconds)
        self._blink_timestamps = [t for t in self._blink_timestamps 
                                   if current_time - t < 60]
        
        # Calculate blinks per minute
        if self._blink_timestamps:
            duration = current_time - self._blink_timestamps[0]
            if duration > 0:
                return len(self._blink_timestamps) * (60 / duration)
        return 0.0
    
    def _calculate_alert_level(self, is_closed: bool, duration_ms: float) -> AlertLevel:
        """Determine alert level based on eye closure duration"""
        if not is_closed:
            return AlertLevel.NONE
        
        if duration_ms >= self.CRITICAL_DURATION_MS:
            return AlertLevel.CRITICAL
        elif duration_ms >= self.ALERT_DURATION_MS:
            return AlertLevel.ALERT
        elif duration_ms >= self.WARNING_DURATION_MS:
            return AlertLevel.WARNING
        
        return AlertLevel.NONE
    
    def process_frame(self, frame: np.ndarray) -> EyeDetectionState:
        """
        Process a video frame and return detection state.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            EyeDetectionState with all detection results
        """
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        # Default state for no detection
        if not self._enabled:
            return self._empty_state(current_time)
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            # No face - decay closed duration
            if self._closed_start:
                self._closed_start = None
            return self._empty_state(current_time, face_detected=False)
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR
        ear_left = self._calculate_ear(landmarks, self.LEFT_EYE_EAR, frame.shape)
        ear_right = self._calculate_ear(landmarks, self.RIGHT_EYE_EAR, frame.shape)
        ear_avg = (ear_left + ear_right) / 2
        
        # Extract eyes for CNN
        eye_img, eye_bboxes = self._extract_eyes(frame, landmarks)
        
        # Classify with CNN (or use EAR as fallback)
        if eye_img is not None and self._model is not None:
            prediction = self._classify_eyes(eye_img)
            cnn_closed = prediction["class"] == "closed"
            confidence = prediction["confidence"]
        else:
            cnn_closed = ear_avg < self.EAR_THRESHOLD
            confidence = 0.7 if cnn_closed else 0.8
        
        # Smooth predictions
        self._prediction_history.append("closed" if cnn_closed else "open")
        if len(self._prediction_history) > self._history_size:
            self._prediction_history.pop(0)
        
        closed_votes = sum(1 for h in self._prediction_history if h == "closed")
        smoothed_closed = closed_votes > len(self._prediction_history) / 2
        
        # Combine CNN + EAR
        ear_closed = ear_avg < self.EAR_THRESHOLD
        is_closed = smoothed_closed or (ear_closed and confidence < 0.8)
        
        # Track blinks (transition from closed to open)
        if self._last_eye_state == "closed" and not is_closed:
            self._blink_count += 1
            self._blink_timestamps.append(current_time)
        self._last_eye_state = "closed" if is_closed else "open"
        
        # Track closed duration
        if is_closed:
            if self._closed_start is None:
                self._closed_start = current_time
            self._closed_duration_ms = (current_time - self._closed_start) * 1000
        else:
            self._closed_start = None
            self._closed_duration_ms = 0.0
        
        # Calculate alert level
        alert_level = self._calculate_alert_level(is_closed, self._closed_duration_ms)
        is_drowsy = alert_level in [AlertLevel.ALERT, AlertLevel.CRITICAL]
        
        # Update session stats
        if is_drowsy:
            self._total_closed_time += dt
            if not self._last_drowsy:
                self._drowsiness_events += 1
        self._last_drowsy = is_drowsy
        
        # Get head pose
        pitch, yaw, roll = self._get_head_pose(landmarks, frame.shape)
        
        # Get landmarks for visualization
        viz_landmarks = self._get_eye_landmarks_for_viz(landmarks, frame.shape)
        
        # Blinks per minute
        bpm = self._update_blink_stats(current_time)
        
        return EyeDetectionState(
            eye_state="closed" if is_closed else "open",
            eyes_closed=is_closed,
            confidence=confidence,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_average=ear_avg,
            closed_duration_ms=self._closed_duration_ms,
            blink_count=self._blink_count,
            blinks_per_minute=bpm,
            alert_level=alert_level,
            is_drowsy=is_drowsy,
            face_detected=True,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            landmarks=viz_landmarks,
            eye_bboxes=eye_bboxes,
            detection_method="eye_classifier" if self._model else "ear_only",
            timestamp=current_time
        )
    
    def _empty_state(self, current_time: float, face_detected: bool = False) -> EyeDetectionState:
        """Return empty state when no detection"""
        return EyeDetectionState(
            eye_state="unknown",
            eyes_closed=False,
            confidence=0.0,
            ear_left=0.0,
            ear_right=0.0,
            ear_average=0.0,
            closed_duration_ms=0.0,
            blink_count=self._blink_count,
            blinks_per_minute=0.0,
            alert_level=AlertLevel.NONE,
            is_drowsy=False,
            face_detected=face_detected,
            detection_method="none",
            timestamp=current_time
        )
    
    def process_base64(self, base64_data: str) -> EyeDetectionState:
        """Process a base64-encoded image"""
        if not CV2_AVAILABLE:
            return self._empty_state(time.time())
        
        try:
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
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
            "duration_seconds": round(duration, 1),
            "total_drowsy_seconds": round(self._total_closed_time, 1),
            "total_distracted_seconds": 0.0,
            "drowsiness_events": self._drowsiness_events,
            "distraction_events": 0,
            "blink_count": self._blink_count
        }
    
    def reset(self):
        """Reset all tracking state"""
        self._closed_start = None
        self._closed_duration_ms = 0.0
        self._blink_count = 0
        self._blink_timestamps = []
        self._last_eye_state = "open"
        self._prediction_history = []
        self._session_start = time.time()
        self._total_closed_time = 0.0
        self._drowsiness_events = 0
        self._last_drowsy = False
        self._last_update = time.time()
        logger.info("EyeDrowsinessDetector reset")


# Singleton instance
_eye_detector: Optional[EyeDrowsinessDetector] = None


def get_eye_detector(model_path: str = "models/eye_classifier.pth") -> EyeDrowsinessDetector:
    """Get or create the singleton EyeDrowsinessDetector"""
    global _eye_detector
    if _eye_detector is None:
        _eye_detector = EyeDrowsinessDetector(model_path=model_path)
    return _eye_detector


def is_eye_detector_available() -> bool:
    """Check if eye detector dependencies are available"""
    return TORCH_AVAILABLE and CV2_AVAILABLE and MEDIAPIPE_AVAILABLE
