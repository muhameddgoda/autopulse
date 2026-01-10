"""
AutoPulse Eye-Based Drowsiness Detector
=======================================
Production-ready detector that integrates with the FastAPI backend.

Combines:
1. MediaPipe for face detection and eye landmark extraction
2. Trained CNN for eye state classification (open/closed)
3. EAR as backup confirmation
4. Head pose for distraction detection
5. Yawning detection via MAR
6. Temporal smoothing for stable results

This replaces the old full-face approach with accurate eye-only detection.
"""

print("=" * 50)
print("[EYE_DETECTOR MODULE] Loading eye_detector.py...")
print("=" * 50)

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
    print(f"[EYE_DETECTOR MODULE] PyTorch loaded: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import cv2
    CV2_AVAILABLE = True
    print(f"[EYE_DETECTOR MODULE] OpenCV loaded: {cv2.__version__}")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")

# MediaPipe import - handle both old (solutions) and new (tasks) API
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_API = None  # "solutions" or "tasks"
mp = None

try:
    import mediapipe as _mp
    mp = _mp
    # Check which API is available
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_API = "solutions"
        print(f"[EYE_DETECTOR MODULE] MediaPipe loaded (legacy solutions API)")
    elif hasattr(mp, 'tasks') and hasattr(mp.tasks, 'vision') and hasattr(mp.tasks.vision, 'FaceLandmarker'):
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_API = "tasks"
        print(f"[EYE_DETECTOR MODULE] MediaPipe loaded (new tasks API, version {mp.__version__})")
    else:
        logger.warning(f"MediaPipe {mp.__version__} loaded but no suitable face detection API found")
        print(f"[EYE_DETECTOR MODULE] MediaPipe loaded but API not compatible")
except ImportError:
    logger.warning("MediaPipe not available")
    print(f"[EYE_DETECTOR MODULE] MediaPipe not installed")

print(f"[EYE_DETECTOR MODULE] TORCH={TORCH_AVAILABLE}, CV2={CV2_AVAILABLE}, MP={MEDIAPIPE_AVAILABLE}, API={MEDIAPIPE_API}")


class AlertLevel(Enum):
    """Alert severity levels"""
    NONE = "none"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


class DistractionType(Enum):
    """Types of distraction"""
    NONE = "none"
    LOOKING_DOWN = "looking_down"
    LOOKING_LEFT = "looking_left"
    LOOKING_RIGHT = "looking_right"
    LOOKING_UP = "looking_up"


@dataclass
class YawnMetrics:
    """Yawning metrics"""
    mar: float  # Mouth Aspect Ratio
    is_yawning: bool
    yawn_duration_ms: float
    yawn_count: int
    yawns_per_minute: float
    fatigue_level: str  # low, moderate, high
    
    def to_dict(self) -> Dict:
        return {
            "mar": round(self.mar, 3),
            "is_yawning": self.is_yawning,
            "yawn_duration_ms": round(self.yawn_duration_ms, 1),
            "yawn_count": self.yawn_count,
            "yawns_per_minute": round(self.yawns_per_minute, 2),
            "fatigue_level": self.fatigue_level
        }


@dataclass
class DistractionMetrics:
    """Head pose distraction metrics"""
    pitch: float
    yaw: float
    roll: float
    is_distracted: bool
    distraction_type: str
    distraction_duration_ms: float
    looking_at_road: bool
    attention_score: float
    
    def to_dict(self) -> Dict:
        return {
            "pitch": round(self.pitch, 1),
            "yaw": round(self.yaw, 1),
            "roll": round(self.roll, 1),
            "is_distracted": self.is_distracted,
            "distraction_type": self.distraction_type,
            "distraction_duration_ms": round(self.distraction_duration_ms, 1),
            "looking_at_road": self.looking_at_road,
            "attention_score": round(self.attention_score, 1)
        }


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
    
    # Yawning metrics (NEW)
    yawn: Optional[YawnMetrics] = None
    
    # Distraction metrics (NEW)
    distraction: Optional[DistractionMetrics] = None
    
    # Visualization data
    landmarks: Optional[Dict] = None
    eye_bboxes: Optional[tuple] = None
    
    # Metadata
    detection_method: str = "eye_classifier"
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        result = {
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
        # Add yawn metrics if present
        if self.yawn:
            result["yawn"] = self.yawn.to_dict()
        # Add distraction metrics if present
        if self.distraction:
            result["distraction"] = self.distraction.to_dict()
        return result

# ============================================================================
# CNN Models (must match training architecture)
# ============================================================================


# Only define these classes if PyTorch is available
if TORCH_AVAILABLE:

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
    
    Features:
    - Eye state detection (CNN + EAR)
    - Yawning detection (MAR algorithm)
    - Head pose distraction detection
    - Blink rate monitoring
    
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
    
    # Mouth landmarks for yawn detection (MAR)
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_MID = [37, 0, 267]
    LOWER_LIP_MID = [84, 17, 314]
    
    # Thresholds - Tuned for better sensitivity
    EAR_THRESHOLD = 0.26  # Increased from 0.22 for better detection
    WARNING_DURATION_MS = 300  # Faster warning
    ALERT_DURATION_MS = 1500  # Faster alert
    CRITICAL_DURATION_MS = 3000  # Faster critical
    
    # Yawn thresholds - More sensitive
    MAR_THRESHOLD = 0.5  # Lowered from 0.6
    YAWN_DURATION_MS = 1200  # Reduced from 1500
    YAWN_COOLDOWN_MS = 1500  # Reduced from 2000
    
    # Head pose thresholds (distraction) - More sensitive
    PITCH_DOWN_THRESHOLD = 15.0  # Looking down (was 20)
    PITCH_UP_THRESHOLD = -20.0  # Looking up (was -25)
    YAW_THRESHOLD = 25.0  # Looking left/right (was 30)
    DISTRACTION_ALERT_MS = 1500  # 1.5 seconds (was 2)
    
    # Image sizes (must match training)
    IMG_HEIGHT = 64
    IMG_WIDTH = 128
    
    def __init__(self, model_path: str="models/eye_classifier.pth"):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained eye classifier model
        """
        self._enabled = False
        self._model = None
        self._device = None
        self._classes = ["closed", "open"]
        self._face_mesh = None
        self._transform = None
        
        print(f"[CV INIT] Starting EyeDrowsinessDetector init...")
        print(f"[CV INIT] TORCH={TORCH_AVAILABLE}, CV2={CV2_AVAILABLE}, MP={MEDIAPIPE_AVAILABLE}")
        
        # MediaPipe is required - model is optional (EAR-only mode)
        if CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
            try:
                # MediaPipe is essential for face/eye detection
                print(f"[CV INIT] Initializing MediaPipe...")
                self._init_mediapipe()
                print(f"[CV INIT] MediaPipe initialized successfully!")
                
                # Enable detector - we can work with EAR-only mode
                self._enabled = True
                
                # Model loading is optional - enhances accuracy but not required
                if TORCH_AVAILABLE:
                    try:
                        print(f"[CV INIT] Initializing model (optional)...")
                        self._init_model(model_path)
                        print(f"[CV INIT] Initializing transforms...")
                        self._init_transforms()
                    except Exception as model_err:
                        print(f"[CV INIT] Model init failed (non-fatal): {model_err}")
                        logger.warning(f"Model initialization failed, using EAR-only mode: {model_err}")
                else:
                    print(f"[CV INIT] PyTorch not available, using EAR-only mode")
                
                print(f"[CV INIT] SUCCESS! Detector enabled (model={'loaded' if self._model else 'EAR-only'})")
                logger.info(f"EyeDrowsinessDetector initialized (mode={'CNN+EAR' if self._model else 'EAR-only'})")
                
            except Exception as e:
                print(f"[CV INIT] FAILED: {e}")
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to initialize detector: {e}")
        else:
            missing = []
            if not CV2_AVAILABLE:
                missing.append("OpenCV")
            if not MEDIAPIPE_AVAILABLE:
                missing.append("MediaPipe")
            print(f"[CV INIT] Missing required dependencies: {missing}")
            logger.error(f"Missing dependencies for EyeDrowsinessDetector: {missing}")
        
        # State tracking - faster response
        self._closed_start: Optional[float] = None
        self._closed_duration_ms = 0.0
        self._blink_count = 0
        self._blink_timestamps: List[float] = []
        self._last_eye_state = "open"
        self._prediction_history: List[str] = []
        self._history_size = 3  # Reduced from 5 for faster detection
        
        # Yawn tracking (NEW)
        self._yawn_open_start: Optional[float] = None
        self._yawn_count = 0
        self._yawn_timestamps: List[float] = []
        self._last_yawn_end: float = 0
        self._mar_history: List[float] = []
        
        # Distraction tracking (NEW)
        self._distraction_start: Optional[float] = None
        self._attention_history: List[float] = []
        self._distraction_events = 0
        self._total_distracted_time = 0.0
        self._last_distracted = False
        
        # Session stats
        self._session_start = time.time()
        self._total_closed_time = 0.0
        self._drowsiness_events = 0
        self._last_drowsy = False
        self._last_update = time.time()
    
    def _init_model(self, model_path: str):
        """Load the trained model with robust path resolution"""
        import os
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try multiple model paths
        candidate_paths = []
        
        # 1. Exact path provided
        if os.path.isabs(model_path):
            candidate_paths.append(model_path)
        
        # 2. Relative to backend directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate_paths.append(os.path.join(base_dir, model_path))
        
        # 3. Relative to backend/models directory
        candidate_paths.append(os.path.join(base_dir, "models", os.path.basename(model_path)))
        
        # 4. Relative to current working directory
        candidate_paths.append(os.path.join(os.getcwd(), model_path))
        candidate_paths.append(os.path.join(os.getcwd(), "models", os.path.basename(model_path)))
        candidate_paths.append(os.path.join(os.getcwd(), "backend", "models", os.path.basename(model_path)))
        
        # 5. Try looking for MobileNet model as alternative
        for base in [base_dir, os.getcwd()]:
            candidate_paths.append(os.path.join(base, "models", "drowsiness_mobilenetv2.pth"))
            candidate_paths.append(os.path.join(base, "backend", "models", "drowsiness_mobilenetv2.pth"))
        
        # Find first existing model
        actual_path = None
        for path in candidate_paths:
            print(f"[CV DEBUG] Checking model path: {path}")
            if os.path.exists(path):
                actual_path = path
                break
        
        if not actual_path:
            logger.warning(f"No model found in any location. Using EAR-only mode.")
            print(f"[CV DEBUG] Model NOT found! Will use EAR-only mode (still functional)")
            print(f"[CV DEBUG] Searched paths: {candidate_paths[:5]}...")
            return
        
        print(f"[CV DEBUG] Model found at: {actual_path}")
        
        try:
            checkpoint = torch.load(actual_path, map_location=self._device, weights_only=False)
            
            model_type = checkpoint.get("model_type", "custom")
            if model_type == "mobilenet" or "mobilenet" in actual_path.lower():
                self._model = EyeClassifierMobileNet()
            else:
                self._model = EyeClassifierCNN()
            
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model = self._model.to(self._device)
            self._model.eval()
            
            self._classes = checkpoint.get("classes", ["closed", "open"])
            
            logger.info(f"Loaded eye classifier from {actual_path}")
            logger.info(f"  Model type: {model_type}")
            logger.info(f"  Classes: {self._classes}")
            logger.info(f"  Device: {self._device}")
            print(f"[CV DEBUG] Model loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using EAR-only mode.")
            print(f"[CV DEBUG] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh with optimized settings - handles both APIs"""
        if MEDIAPIPE_API == "solutions":
            # Legacy solutions API (MediaPipe < 0.10)
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                static_image_mode=False
            )
            self._mediapipe_mode = "solutions"
            
        elif MEDIAPIPE_API == "tasks":
            # New Tasks API (MediaPipe >= 0.10)
            import os
            base_options = mp.tasks.BaseOptions(
                model_asset_path=self._get_face_landmarker_model()
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self._face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._mediapipe_mode = "tasks"
        else:
            raise ValueError(f"Unknown MediaPipe API: {MEDIAPIPE_API}")
    
    def _get_face_landmarker_model(self) -> str:
        """Get or download the face landmarker model for Tasks API"""
        import os
        import urllib.request
        
        # Model paths to check
        model_filename = "face_landmarker_v2_with_blendshapes.task"
        candidate_paths = [
            os.path.join(os.getcwd(), "models", model_filename),
            os.path.join(os.getcwd(), "backend", "models", model_filename),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", model_filename),
        ]
        
        for path in candidate_paths:
            if os.path.exists(path):
                print(f"[CV INIT] Found FaceLandmarker model at: {path}")
                return path
        
        # Download if not found
        download_path = candidate_paths[0]
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"[CV INIT] Downloading FaceLandmarker model to {download_path}...")
        
        try:
            urllib.request.urlretrieve(model_url, download_path)
            print(f"[CV INIT] FaceLandmarker model downloaded successfully!")
            return download_path
        except Exception as e:
            raise RuntimeError(f"Failed to download FaceLandmarker model: {e}")
    
    def _get_face_landmarks(self, rgb_frame: np.ndarray):
        """
        Get face landmarks from RGB frame - handles both MediaPipe APIs.
        
        Args:
            rgb_frame: RGB image (not BGR!)
            
        Returns:
            Landmarks list or None if no face detected
        """
        if self._mediapipe_mode == "solutions":
            # Legacy solutions API
            results = self._face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None
            
        elif self._mediapipe_mode == "tasks":
            # New Tasks API - needs mp.Image wrapper
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self._face_mesh.detect(mp_image)
            
            if results.face_landmarks and len(results.face_landmarks) > 0:
                # Convert to same format as solutions API (normalized coords with x, y, z)
                landmarks = results.face_landmarks[0]
                return landmarks  # Already NormalizedLandmark objects
            return None
        
        return None
    
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
    
    def _calculate_mar(self, landmarks, shape) -> float:
        """Calculate Mouth Aspect Ratio for yawn detection"""
        h, w = shape[:2]
        
        # Get mouth points
        left = np.array([landmarks[self.MOUTH_LEFT].x * w, landmarks[self.MOUTH_LEFT].y * h])
        right = np.array([landmarks[self.MOUTH_RIGHT].x * w, landmarks[self.MOUTH_RIGHT].y * h])
        top = np.array([landmarks[self.MOUTH_TOP].x * w, landmarks[self.MOUTH_TOP].y * h])
        bottom = np.array([landmarks[self.MOUTH_BOTTOM].x * w, landmarks[self.MOUTH_BOTTOM].y * h])
        
        # Additional vertical points
        upper_pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in self.UPPER_LIP_MID]
        lower_pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in self.LOWER_LIP_MID]
        
        # Horizontal distance
        horizontal = np.linalg.norm(right - left)
        if horizontal < 1e-6:
            return 0.0
        
        # Vertical distances
        vertical_main = np.linalg.norm(bottom - top)
        vertical_left = np.linalg.norm(lower_pts[0] - upper_pts[0])
        vertical_right = np.linalg.norm(lower_pts[2] - upper_pts[2])
        vertical_avg = (vertical_main + vertical_left + vertical_right) / 3
        
        mar = vertical_avg / horizontal
        
        # Smooth MAR
        self._mar_history.append(mar)
        if len(self._mar_history) > 3:
            self._mar_history.pop(0)
        
        return np.mean(self._mar_history)
    
    def _process_yawn(self, mar: float, current_time: float) -> YawnMetrics:
        """Process yawn detection from MAR"""
        mouth_open = mar > self.MAR_THRESHOLD
        yawn_duration_ms = 0.0
        is_yawning = False
        
        if mouth_open:
            if self._yawn_open_start is None:
                self._yawn_open_start = current_time
            yawn_duration_ms = (current_time - self._yawn_open_start) * 1000
            
            if yawn_duration_ms >= self.YAWN_DURATION_MS:
                is_yawning = True
        else:
            # Mouth closed - check if we finished a yawn
            if self._yawn_open_start is not None:
                final_duration = (current_time - self._yawn_open_start) * 1000
                if final_duration >= self.YAWN_DURATION_MS:
                    time_since_last = (current_time - self._last_yawn_end) * 1000
                    if time_since_last >= self.YAWN_COOLDOWN_MS:
                        self._yawn_count += 1
                        self._yawn_timestamps.append(current_time)
                        self._last_yawn_end = current_time
            self._yawn_open_start = None
        
        # Calculate yawns per minute (over last 5 minutes)
        cutoff = current_time - 300
        recent_yawns = [t for t in self._yawn_timestamps if t > cutoff]
        if recent_yawns:
            time_span = min(300, current_time - recent_yawns[0])
            yawns_per_minute = (len(recent_yawns) / time_span) * 60 if time_span > 0 else 0
        else:
            yawns_per_minute = 0.0
        
        # Fatigue level based on yawns per 5 minutes
        yawns_per_5min = yawns_per_minute * 5
        if yawns_per_5min >= 3:
            fatigue_level = "high"
        elif yawns_per_5min >= 2:
            fatigue_level = "moderate"
        else:
            fatigue_level = "low"
        
        return YawnMetrics(
            mar=mar,
            is_yawning=is_yawning,
            yawn_duration_ms=yawn_duration_ms,
            yawn_count=self._yawn_count,
            yawns_per_minute=yawns_per_minute,
            fatigue_level=fatigue_level
        )
    
    def _process_distraction(self, pitch: float, yaw: float, roll: float, current_time: float, dt: float) -> DistractionMetrics:
        """Process head pose for distraction detection"""
        # Classify distraction type
        distraction_type = DistractionType.NONE
        if pitch > self.PITCH_DOWN_THRESHOLD:
            distraction_type = DistractionType.LOOKING_DOWN
        elif pitch < self.PITCH_UP_THRESHOLD:
            distraction_type = DistractionType.LOOKING_UP
        elif yaw < -self.YAW_THRESHOLD:
            distraction_type = DistractionType.LOOKING_LEFT
        elif yaw > self.YAW_THRESHOLD:
            distraction_type = DistractionType.LOOKING_RIGHT
        
        looking_at_road = distraction_type == DistractionType.NONE
        
        # Track distraction duration
        distraction_duration_ms = 0.0
        if not looking_at_road:
            if self._distraction_start is None:
                self._distraction_start = current_time
            distraction_duration_ms = (current_time - self._distraction_start) * 1000
        else:
            self._distraction_start = None
        
        # Is distracted if looking away for too long
        is_distracted = distraction_duration_ms >= self.DISTRACTION_ALERT_MS
        
        # Update stats
        if is_distracted:
            self._total_distracted_time += dt
            if not self._last_distracted:
                self._distraction_events += 1
        self._last_distracted = is_distracted
        
        # Attention score (0-100)
        self._attention_history.append(1.0 if looking_at_road else 0.0)
        if len(self._attention_history) > 30:
            self._attention_history.pop(0)
        attention_score = (sum(self._attention_history) / len(self._attention_history)) * 100 if self._attention_history else 100
        
        return DistractionMetrics(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            is_distracted=is_distracted,
            distraction_type=distraction_type.value,
            distraction_duration_ms=distraction_duration_ms,
            looking_at_road=looking_at_road,
            attention_score=attention_score
        )

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
            print(f"[CV DEBUG] Detector not enabled! TORCH={TORCH_AVAILABLE}, CV2={CV2_AVAILABLE}, MP={MEDIAPIPE_AVAILABLE}")
            return self._empty_state(current_time)
        
        # Check if face_mesh is initialized
        if self._face_mesh is None:
            print(f"[CV DEBUG] Face mesh not initialized!")
            return self._empty_state(current_time)
        
        # Process with MediaPipe - handle both APIs
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self._get_face_landmarks(rgb)
        
        if landmarks is None:
            # No face - decay closed duration
            if self._closed_start:
                self._closed_start = None
            return self._empty_state(current_time, face_detected=False)
        
        # Calculate EAR
        ear_left = self._calculate_ear(landmarks, self.LEFT_EYE_EAR, frame.shape)
        ear_right = self._calculate_ear(landmarks, self.RIGHT_EYE_EAR, frame.shape)
        ear_avg = (ear_left + ear_right) / 2
        
        # Extract eyes for CNN
        eye_img, eye_bboxes = self._extract_eyes(frame, landmarks)
        
        # Classify with CNN (or use EAR as fallback)
        if eye_img is not None and self._model is not None and self._transform is not None:
            prediction = self._classify_eyes(eye_img)
            cnn_closed = prediction["class"] == "closed"
            confidence = prediction["confidence"]
        else:
            # EAR-only mode - use EAR threshold directly
            cnn_closed = ear_avg < self.EAR_THRESHOLD
            confidence = 0.85 if cnn_closed else 0.9  # Higher confidence for EAR-only
        
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
        
        # ===== NEW: Yawn detection =====
        mar = self._calculate_mar(landmarks, frame.shape)
        yawn_metrics = self._process_yawn(mar, current_time)
        
        # ===== NEW: Distraction detection =====
        distraction_metrics = self._process_distraction(pitch, yaw, roll, current_time, dt)
        
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
            yawn=yawn_metrics,
            distraction=distraction_metrics,
            landmarks=viz_landmarks,
            eye_bboxes=eye_bboxes,
            detection_method="eye_classifier" if self._model else "ear_only",
            timestamp=current_time
        )
    
    def _empty_state(self, current_time: float, face_detected: bool=False) -> EyeDetectionState:
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
        
        # Calculate yawns per minute for session
        yawns_per_minute = (self._yawn_count / (duration / 60)) if duration > 0 else 0
        
        return {
            "duration_seconds": round(duration, 1),
            "total_drowsy_seconds": round(self._total_closed_time, 1),
            "total_distracted_seconds": round(self._total_distracted_time, 1),
            "drowsiness_events": self._drowsiness_events,
            "distraction_events": self._distraction_events,
            "blink_count": self._blink_count,
            "yawn_count": self._yawn_count,
            "yawns_per_minute": round(yawns_per_minute, 2)
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
        # Reset yawn tracking
        self._yawn_open_start = None
        self._yawn_count = 0
        self._yawn_timestamps = []
        self._last_yawn_end = 0
        self._mar_history = []
        # Reset distraction tracking
        self._distraction_start = None
        self._attention_history = []
        self._distraction_events = 0
        self._total_distracted_time = 0.0
        self._last_distracted = False
        logger.info("EyeDrowsinessDetector reset")


# Singleton instance
_eye_detector: Optional[EyeDrowsinessDetector] = None


def get_eye_detector(model_path: str="models/eye_classifier.pth") -> EyeDrowsinessDetector:
    """Get or create the singleton EyeDrowsinessDetector"""
    global _eye_detector
    if _eye_detector is None:
        print(f"[CV SINGLETON] Creating new EyeDrowsinessDetector instance...")
        _eye_detector = EyeDrowsinessDetector(model_path=model_path)
        print(f"[CV SINGLETON] Detector created, enabled={_eye_detector.enabled}")
    return _eye_detector


def reset_eye_detector():
    """Reset the singleton detector (useful for reinitializing after errors)"""
    global _eye_detector
    _eye_detector = None
    print(f"[CV SINGLETON] Detector reset, will reinitialize on next call")


def is_eye_detector_available() -> bool:
    """Check if eye detector dependencies are available"""
    return TORCH_AVAILABLE and CV2_AVAILABLE and MEDIAPIPE_AVAILABLE
