"""
AutoPulse Yawn Detector
=======================
Detects yawning using Mouth Aspect Ratio (MAR) algorithm with MediaPipe Face Mesh.

Similar to EAR for eyes, MAR calculates the openness of the mouth:
    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)

When the mouth opens wide (yawn), the vertical distances increase significantly,
causing MAR to spike. Yawning indicates fatigue and is a precursor to drowsiness.

Thresholds:
- MAR > 0.6 for > 1.5 seconds = Yawn detected
- > 3 yawns in 5 minutes = High fatigue warning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class YawnState(Enum):
    """Yawn detection states"""
    NONE = "none"
    MOUTH_OPEN = "mouth_open"
    YAWNING = "yawning"


@dataclass
class YawnMetrics:
    """Yawning detection metrics"""
    mar: float  # Mouth Aspect Ratio
    is_yawning: bool  # Currently yawning
    yawn_duration_ms: float  # Current yawn duration
    yawn_count: int  # Total yawns this session
    yawns_per_minute: float  # Yawn rate (fatigue indicator)
    fatigue_level: str  # "low", "moderate", "high"
    state: YawnState
    
    def to_dict(self) -> Dict:
        return {
            "mar": round(self.mar, 3),
            "is_yawning": self.is_yawning,
            "yawn_duration_ms": round(self.yawn_duration_ms, 1),
            "yawn_count": self.yawn_count,
            "yawns_per_minute": round(self.yawns_per_minute, 2),
            "fatigue_level": self.fatigue_level,
            "state": self.state.value
        }


class YawnDetector:
    """
    Detects yawning using Mouth Aspect Ratio (MAR).
    
    MediaPipe Face Mesh mouth landmarks are used to calculate MAR.
    A sustained high MAR indicates yawning, which is a fatigue indicator.
    
    Detection Logic:
    - MAR > 0.6 = Mouth open (potential yawn)
    - MAR > 0.6 for > 1.5s = Yawn confirmed
    - > 3 yawns in 5 min = High fatigue
    - > 2 yawns in 5 min = Moderate fatigue
    """
    
    # MediaPipe Face Mesh mouth landmark indices
    # Outer lips
    UPPER_LIP_TOP = [13]  # Top of upper lip
    LOWER_LIP_BOTTOM = [14]  # Bottom of lower lip
    LEFT_MOUTH_CORNER = [61]  # Left corner
    RIGHT_MOUTH_CORNER = [291]  # Right corner
    
    # Inner lips (more accurate for MAR)
    UPPER_INNER_LIP = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    LOWER_INNER_LIP = [14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87]
    
    # Simplified 8-point mouth for MAR calculation
    # [left_corner, upper_left, upper_mid, upper_right, right_corner, lower_right, lower_mid, lower_left]
    MOUTH_INDICES = [61, 40, 13, 270, 291, 321, 14, 91]
    
    # Alternative simpler indices
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_MID = [37, 0, 267]
    LOWER_LIP_MID = [84, 17, 314]
    
    # Thresholds - More sensitive
    MAR_THRESHOLD = 0.5  # Above this = mouth open (lowered from 0.6)
    YAWN_DURATION_MS = 1200  # Min duration to count as yawn (reduced from 1500)
    YAWN_COOLDOWN_MS = 1500  # Cooldown between yawns (reduced from 2000)
    
    # Fatigue thresholds (yawns per 5 minutes) - More sensitive
    HIGH_FATIGUE_YAWNS = 2  # Was 3
    MODERATE_FATIGUE_YAWNS = 1  # Was 2
    
    def __init__(self, mar_threshold: float=0.6):
        """Initialize the yawn detector"""
        self.mar_threshold = mar_threshold
        
        # State tracking
        self._mouth_open_start: Optional[float] = None
        self._yawn_timestamps: List[float] = []
        self._yawn_count: int = 0
        self._last_yawn_end: float = 0
        self._current_state: YawnState = YawnState.NONE
        
        # For smoothing
        self._mar_history: List[float] = []
        self._history_size = 3
        
    def calculate_mar(
        self,
        landmarks: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate Mouth Aspect Ratio.
        
        MAR = (vertical distances) / (horizontal distance)
        
        Args:
            landmarks: All facial landmarks as (x, y) tuples
            
        Returns:
            Mouth Aspect Ratio value
        """
        # Get key mouth points
        left_corner = np.array(landmarks[self.MOUTH_LEFT])
        right_corner = np.array(landmarks[self.MOUTH_RIGHT])
        top = np.array(landmarks[self.MOUTH_TOP])
        bottom = np.array(landmarks[self.MOUTH_BOTTOM])
        
        # Get additional vertical points for accuracy
        upper_points = [np.array(landmarks[i]) for i in self.UPPER_LIP_MID]
        lower_points = [np.array(landmarks[i]) for i in self.LOWER_LIP_MID]
        
        # Calculate horizontal distance
        horizontal = np.linalg.norm(right_corner - left_corner)
        
        if horizontal < 1e-6:
            return 0.0
        
        # Calculate vertical distances
        vertical_main = np.linalg.norm(bottom - top)
        
        # Additional vertical measurements
        vertical_left = np.linalg.norm(lower_points[0] - upper_points[0])
        vertical_right = np.linalg.norm(lower_points[2] - upper_points[2])
        
        # Average vertical distance
        vertical_avg = (vertical_main + vertical_left + vertical_right) / 3
        
        # Calculate MAR
        mar = vertical_avg / horizontal
        
        return mar
    
    def _smooth_mar(self, mar: float) -> float:
        """Apply smoothing to MAR value"""
        self._mar_history.append(mar)
        if len(self._mar_history) > self._history_size:
            self._mar_history.pop(0)
        return np.mean(self._mar_history)
    
    def _calculate_yawns_per_minute(self) -> float:
        """Calculate yawn rate over last 5 minutes"""
        current_time = time.time()
        cutoff = current_time - 300  # 5 minutes
        
        recent_yawns = [t for t in self._yawn_timestamps if t > cutoff]
        
        if len(recent_yawns) < 1:
            return 0.0
        
        # Calculate rate
        time_span = min(300, current_time - recent_yawns[0]) if recent_yawns else 300
        if time_span > 0:
            return (len(recent_yawns) / time_span) * 60
        return 0.0
    
    def _determine_fatigue_level(self, yawns_per_minute: float) -> str:
        """Determine fatigue level based on yawn rate"""
        # Convert to yawns per 5 minutes for comparison
        yawns_per_5min = yawns_per_minute * 5
        
        if yawns_per_5min >= self.HIGH_FATIGUE_YAWNS:
            return "high"
        elif yawns_per_5min >= self.MODERATE_FATIGUE_YAWNS:
            return "moderate"
        return "low"
    
    def process(
        self,
        landmarks: List[Tuple[float, float]]
    ) -> YawnMetrics:
        """
        Process facial landmarks and detect yawning.
        
        Args:
            landmarks: Facial landmarks from MediaPipe
            
        Returns:
            YawnMetrics with detection results
        """
        current_time = time.time()
        
        # Calculate MAR
        mar_raw = self.calculate_mar(landmarks)
        mar = self._smooth_mar(mar_raw)
        
        # Determine mouth state
        mouth_open = mar > self.mar_threshold
        
        # Track yawn duration
        yawn_duration_ms = 0.0
        is_yawning = False
        state = YawnState.NONE
        
        if mouth_open:
            state = YawnState.MOUTH_OPEN
            
            if self._mouth_open_start is None:
                self._mouth_open_start = current_time
            
            yawn_duration_ms = (current_time - self._mouth_open_start) * 1000
            
            # Check if this qualifies as a yawn
            if yawn_duration_ms >= self.YAWN_DURATION_MS:
                is_yawning = True
                state = YawnState.YAWNING
                
        else:
            # Mouth closed - check if we just finished a yawn
            if self._mouth_open_start is not None:
                final_duration = (current_time - self._mouth_open_start) * 1000
                
                # Was it a yawn? Check duration and cooldown
                if final_duration >= self.YAWN_DURATION_MS:
                    time_since_last = (current_time - self._last_yawn_end) * 1000
                    if time_since_last >= self.YAWN_COOLDOWN_MS:
                        # Count this as a yawn
                        self._yawn_count += 1
                        self._yawn_timestamps.append(current_time)
                        self._last_yawn_end = current_time
            
            self._mouth_open_start = None
        
        # Calculate metrics
        yawns_per_minute = self._calculate_yawns_per_minute()
        fatigue_level = self._determine_fatigue_level(yawns_per_minute)
        
        self._current_state = state
        
        return YawnMetrics(
            mar=mar,
            is_yawning=is_yawning,
            yawn_duration_ms=yawn_duration_ms,
            yawn_count=self._yawn_count,
            yawns_per_minute=yawns_per_minute,
            fatigue_level=fatigue_level,
            state=state
        )
    
    def reset(self):
        """Reset all tracking state"""
        self._mouth_open_start = None
        self._yawn_timestamps = []
        self._yawn_count = 0
        self._last_yawn_end = 0
        self._current_state = YawnState.NONE
        self._mar_history = []
    
    def get_summary(self) -> Dict:
        """Get session summary"""
        return {
            "total_yawns": self._yawn_count,
            "yawns_per_minute": round(self._calculate_yawns_per_minute(), 2),
            "fatigue_level": self._determine_fatigue_level(self._calculate_yawns_per_minute())
        }


# Singleton instance
_yawn_detector: Optional[YawnDetector] = None


def get_yawn_detector() -> YawnDetector:
    """Get or create the singleton YawnDetector instance"""
    global _yawn_detector
    if _yawn_detector is None:
        _yawn_detector = YawnDetector()
    return _yawn_detector
