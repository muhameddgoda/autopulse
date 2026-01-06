"""
AutoPulse Eye Tracker
=====================
Detects drowsiness using Eye Aspect Ratio (EAR) algorithm with MediaPipe Face Mesh.

The Eye Aspect Ratio is calculated as:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Where p1-p6 are the eye landmark points. When the eye closes, the vertical
distances decrease, causing EAR to drop. A sustained low EAR indicates drowsiness.

Reference: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    cv2 = None


class AlertLevel(Enum):
    """Drowsiness alert severity levels"""
    NONE = "none"
    WARNING = "warning"      # Eyes closing, might be drowsy
    ALERT = "alert"          # Confirmed drowsiness
    CRITICAL = "critical"    # Prolonged drowsiness - immediate action needed


@dataclass
class EyeMetrics:
    """Metrics for a single eye"""
    ear: float              # Eye Aspect Ratio
    is_open: bool           # Whether eye is considered open
    landmarks: List[Tuple[int, int]]  # Eye landmark pixel coordinates


@dataclass
class DrowsinessState:
    """Current drowsiness detection state"""
    # Eye metrics
    ear_left: float
    ear_right: float
    ear_average: float
    
    # Detection state
    eyes_closed: bool
    closed_duration_ms: float
    blink_count: int
    blinks_per_minute: float
    
    # Alert status
    alert_level: AlertLevel
    is_drowsy: bool
    
    # Head pose (optional)
    head_pitch: Optional[float] = None
    head_yaw: Optional[float] = None
    head_roll: Optional[float] = None
    
    # Face detection
    face_detected: bool = True
    confidence: float = 1.0
    
    # Landmarks for visualization (optional)
    landmarks: Optional[dict] = None
    
    # Timestamps
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "ear_left": float(round(self.ear_left, 3)),
            "ear_right": float(round(self.ear_right, 3)),
            "ear_average": float(round(self.ear_average, 3)),
            "eyes_closed": bool(self.eyes_closed),
            "closed_duration_ms": float(round(self.closed_duration_ms, 1)),
            "blink_count": int(self.blink_count),
            "blinks_per_minute": float(round(self.blinks_per_minute, 1)),
            "alert_level": self.alert_level.value,
            "is_drowsy": bool(self.is_drowsy),
            "head_pitch": float(round(self.head_pitch, 1)) if self.head_pitch is not None else None,
            "head_yaw": float(round(self.head_yaw, 1)) if self.head_yaw is not None else None,
            "head_roll": float(round(self.head_roll, 1)) if self.head_roll is not None else None,
            "face_detected": bool(self.face_detected),
            "confidence": float(round(self.confidence, 2)),
            "timestamp": float(self.timestamp),
        }
        if self.landmarks:
            result["landmarks"] = self.landmarks
        return result


class EyeTracker:
    """
    Real-time drowsiness detection using Eye Aspect Ratio (EAR).
    
    Detection Logic:
    - EAR < threshold for > 2 seconds = DROWSY (ALERT)
    - EAR < threshold for > 0.5 seconds = WARNING
    - EAR < threshold for > 4 seconds = CRITICAL
    - High blink rate (>25/min) = fatigue indicator
    
    MediaPipe Face Mesh provides 468 facial landmarks. We use specific
    indices for the eye contours to calculate EAR.
    """
    
    # MediaPipe Face Mesh landmark indices for eyes
    # These form the eye contour used for EAR calculation
    # Order: [outer corner, upper1, upper2, inner corner, lower1, lower2]
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Additional landmarks for more precise tracking
    LEFT_EYE_FULL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_FULL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Iris landmarks (for gaze direction - future enhancement)
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    
    # Face oval for head pose estimation
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    # Thresholds
    EAR_THRESHOLD = 0.22           # Below this = eyes closed
    EAR_THRESHOLD_LOW = 0.18       # Very closed eyes
    
    # Timing thresholds (milliseconds)
    WARNING_DURATION_MS = 500      # 0.5 seconds
    ALERT_DURATION_MS = 2000       # 2 seconds  
    CRITICAL_DURATION_MS = 4000    # 4 seconds
    
    # Blink detection
    BLINK_DURATION_MAX_MS = 400    # Max duration for a blink (vs drowsiness)
    HIGH_BLINK_RATE = 25           # Blinks per minute indicating fatigue
    
    def __init__(
        self,
        ear_threshold: float = 0.22,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the eye tracker.
        
        Args:
            ear_threshold: EAR below this is considered "closed"
            min_detection_confidence: MediaPipe face detection confidence
            min_tracking_confidence: MediaPipe face tracking confidence
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe and OpenCV are required for drowsiness detection. "
                "Install with: pip install mediapipe opencv-python"
            )
        
        self.ear_threshold = ear_threshold
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Includes iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # State tracking
        self._eyes_closed_start: Optional[float] = None
        self._blink_timestamps: List[float] = []
        self._blink_count: int = 0
        self._last_ear: float = 0.3
        self._was_closed: bool = False
        
        # For smoothing
        self._ear_history: List[float] = []
        self._history_size = 5
        
    def calculate_ear(
        self,
        landmarks: List[Tuple[float, float]],
        eye_indices: List[int]
    ) -> float:
        """
        Calculate Eye Aspect Ratio for given eye landmarks.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            landmarks: All facial landmarks as (x, y) tuples
            eye_indices: Indices of the 6 eye landmarks
            
        Returns:
            Eye Aspect Ratio value
        """
        # Get the 6 eye landmarks
        # [0]: outer corner, [1]: upper outer, [2]: upper inner
        # [3]: inner corner, [4]: lower inner, [5]: lower outer
        points = [landmarks[i] for i in eye_indices]
        
        # Convert to numpy arrays for calculation
        p1 = np.array(points[0])  # Outer corner
        p2 = np.array(points[1])  # Upper outer
        p3 = np.array(points[2])  # Upper inner
        p4 = np.array(points[3])  # Inner corner
        p5 = np.array(points[4])  # Lower inner
        p6 = np.array(points[5])  # Lower outer
        
        # Calculate vertical distances
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        
        # Calculate horizontal distance
        horizontal = np.linalg.norm(p1 - p4)
        
        # Avoid division by zero
        if horizontal < 1e-6:
            return 0.0
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        return ear
    
    def estimate_head_pose(
        self,
        landmarks: List[Tuple[float, float]],
        image_size: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll) from facial landmarks.
        
        Uses a simplified approach based on key facial points.
        
        Args:
            landmarks: Facial landmarks
            image_size: (width, height) of the image
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        w, h = image_size
        
        # Key points for pose estimation
        nose_tip = landmarks[4]
        chin = landmarks[152]
        left_eye_outer = landmarks[263]
        right_eye_outer = landmarks[33]
        left_mouth = landmarks[287]
        right_mouth = landmarks[57]
        
        # Simple yaw estimation (left-right rotation)
        # Based on nose position relative to eye centers
        eye_center_x = (left_eye_outer[0] + right_eye_outer[0]) / 2
        yaw = (nose_tip[0] - eye_center_x) * 100  # Simplified
        
        # Simple pitch estimation (up-down rotation)
        # Based on vertical distance ratios
        eye_center_y = (left_eye_outer[1] + right_eye_outer[1]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        face_height = mouth_center_y - eye_center_y
        nose_ratio = (nose_tip[1] - eye_center_y) / face_height if face_height > 0 else 0.5
        pitch = (nose_ratio - 0.5) * 60  # Simplified
        
        # Simple roll estimation (head tilt)
        eye_diff_y = left_eye_outer[1] - right_eye_outer[1]
        eye_diff_x = left_eye_outer[0] - right_eye_outer[0]
        roll = np.degrees(np.arctan2(eye_diff_y, eye_diff_x))
        
        return pitch, yaw, roll
    
    def _smooth_ear(self, ear: float) -> float:
        """Apply smoothing to EAR value to reduce noise"""
        self._ear_history.append(ear)
        if len(self._ear_history) > self._history_size:
            self._ear_history.pop(0)
        return np.mean(self._ear_history)
    
    def _update_blink_tracking(self, eyes_closed: bool, current_time: float) -> None:
        """Track blinks for fatigue detection"""
        if self._was_closed and not eyes_closed:
            # Eyes just opened - check if it was a blink
            if self._eyes_closed_start:
                closed_duration = (current_time - self._eyes_closed_start) * 1000
                if closed_duration < self.BLINK_DURATION_MAX_MS:
                    # This was a blink, not drowsiness
                    self._blink_timestamps.append(current_time)
                    self._blink_count += 1
        
        self._was_closed = eyes_closed
        
        # Remove old blinks (older than 1 minute)
        cutoff = current_time - 60
        self._blink_timestamps = [t for t in self._blink_timestamps if t > cutoff]
    
    def _calculate_blinks_per_minute(self) -> float:
        """Calculate current blink rate"""
        if len(self._blink_timestamps) < 2:
            return 0.0
        
        # Calculate rate based on recent blinks
        time_span = self._blink_timestamps[-1] - self._blink_timestamps[0]
        if time_span > 0:
            return (len(self._blink_timestamps) / time_span) * 60
        return 0.0
    
    def _determine_alert_level(
        self,
        eyes_closed: bool,
        closed_duration_ms: float,
        blinks_per_minute: float
    ) -> AlertLevel:
        """Determine the current alert level"""
        if not eyes_closed:
            # Check for high blink rate (fatigue indicator)
            if blinks_per_minute > self.HIGH_BLINK_RATE:
                return AlertLevel.WARNING
            return AlertLevel.NONE
        
        # Eyes are closed - check duration
        if closed_duration_ms >= self.CRITICAL_DURATION_MS:
            return AlertLevel.CRITICAL
        elif closed_duration_ms >= self.ALERT_DURATION_MS:
            return AlertLevel.ALERT
        elif closed_duration_ms >= self.WARNING_DURATION_MS:
            return AlertLevel.WARNING
        
        return AlertLevel.NONE
    
    def process_frame(self, frame: np.ndarray) -> DrowsinessState:
        """
        Process a video frame and return drowsiness state.
        
        Args:
            frame: BGR image from camera (numpy array)
            
        Returns:
            DrowsinessState with all detection metrics
        """
        current_time = time.time()
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        # No face detected
        if not results.multi_face_landmarks:
            return DrowsinessState(
                ear_left=0.0,
                ear_right=0.0,
                ear_average=0.0,
                eyes_closed=False,
                closed_duration_ms=0.0,
                blink_count=self._blink_count,
                blinks_per_minute=self._calculate_blinks_per_minute(),
                alert_level=AlertLevel.NONE,
                is_drowsy=False,
                face_detected=False,
                confidence=0.0,
                timestamp=current_time
            )
        
        # Get landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks.landmark
        ]
        
        # Calculate EAR for both eyes
        ear_left = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        ear_right = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        ear_avg = (ear_left + ear_right) / 2
        
        # Smooth the EAR value
        ear_smoothed = self._smooth_ear(ear_avg)
        
        # Determine if eyes are closed
        eyes_closed = ear_smoothed < self.ear_threshold
        
        # Track closed duration
        if eyes_closed:
            if self._eyes_closed_start is None:
                self._eyes_closed_start = current_time
            closed_duration_ms = (current_time - self._eyes_closed_start) * 1000
        else:
            closed_duration_ms = 0.0
            self._eyes_closed_start = None
        
        # Update blink tracking
        self._update_blink_tracking(eyes_closed, current_time)
        blinks_per_minute = self._calculate_blinks_per_minute()
        
        # Determine alert level
        alert_level = self._determine_alert_level(
            eyes_closed, closed_duration_ms, blinks_per_minute
        )
        
        # Estimate head pose
        pitch, yaw, roll = self.estimate_head_pose(landmarks, (w, h))
        
        # Is drowsy if alert level is ALERT or CRITICAL
        is_drowsy = alert_level in [AlertLevel.ALERT, AlertLevel.CRITICAL]
        
        # Extract just the eye landmarks for visualization
        left_eye_pts = [[landmarks[i][0], landmarks[i][1]] for i in self.LEFT_EYE_INDICES]
        right_eye_pts = [[landmarks[i][0], landmarks[i][1]] for i in self.RIGHT_EYE_INDICES]

        return DrowsinessState(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_average=ear_smoothed,
            eyes_closed=eyes_closed,
            closed_duration_ms=closed_duration_ms,
            blink_count=self._blink_count,
            blinks_per_minute=blinks_per_minute,
            alert_level=alert_level,
            is_drowsy=is_drowsy,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            face_detected=True,
            confidence=0.95,
            landmarks={"left_eye": left_eye_pts, "right_eye": right_eye_pts},  # ADD THIS
            timestamp=current_time
        )
    
    def process_base64_frame(self, base64_data: str) -> DrowsinessState:
        """
        Process a base64-encoded image frame.
        
        Args:
            base64_data: Base64 encoded image (with or without data URL prefix)
            
        Returns:
            DrowsinessState
        """
        import base64
        
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return DrowsinessState(
                ear_left=0.0,
                ear_right=0.0,
                ear_average=0.0,
                eyes_closed=False,
                closed_duration_ms=0.0,
                blink_count=0,
                blinks_per_minute=0.0,
                alert_level=AlertLevel.NONE,
                is_drowsy=False,
                face_detected=False,
                confidence=0.0,
                timestamp=time.time()
            )
        
        return self.process_frame(frame)
    
    def reset(self) -> None:
        """Reset all tracking state"""
        self._eyes_closed_start = None
        self._blink_timestamps = []
        self._blink_count = 0
        self._last_ear = 0.3
        self._was_closed = False
        self._ear_history = []
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Singleton instance
_eye_tracker: Optional[EyeTracker] = None


def get_eye_tracker() -> EyeTracker:
    """Get or create the singleton EyeTracker instance"""
    global _eye_tracker
    if _eye_tracker is None:
        _eye_tracker = EyeTracker()
    return _eye_tracker
