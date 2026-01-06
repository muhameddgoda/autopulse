"""
AutoPulse Head Pose Estimator
=============================
Detects driver distraction by analyzing head pose (pitch, yaw, roll).

Distraction is detected when the driver looks away from the road for
too long, indicating they might be looking at their phone, the passenger,
or other distractions.

Thresholds:
- Looking down (phone): pitch > 20°
- Looking left/right: |yaw| > 30°
- Sustained distraction: > 2 seconds
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class DistractionType(Enum):
    """Types of detected distraction"""
    NONE = "none"
    LOOKING_DOWN = "looking_down"       # Phone, dashboard
    LOOKING_LEFT = "looking_left"       # Passenger, window
    LOOKING_RIGHT = "looking_right"     # Passenger, window
    LOOKING_UP = "looking_up"           # Rearview mirror (brief OK)
    HEAD_TILTED = "head_tilted"         # Fatigue indicator


@dataclass
class HeadPose:
    """Head pose angles in degrees"""
    pitch: float  # Up/down (-90 to 90, positive = looking down)
    yaw: float    # Left/right (-90 to 90, positive = looking right)
    roll: float   # Head tilt (-90 to 90, positive = tilting right)
    
    def to_dict(self) -> Dict:
        return {
            "pitch": round(self.pitch, 1),
            "yaw": round(self.yaw, 1),
            "roll": round(self.roll, 1)
        }


@dataclass
class DistractionState:
    """Current distraction detection state"""
    head_pose: HeadPose
    is_distracted: bool
    distraction_type: DistractionType
    distraction_duration_ms: float
    looking_at_road: bool
    attention_score: float  # 0-100, 100 = fully attentive
    
    def to_dict(self) -> Dict:
        return {
            "head_pose": self.head_pose.to_dict(),
            "is_distracted": self.is_distracted,
            "distraction_type": self.distraction_type.value,
            "distraction_duration_ms": round(self.distraction_duration_ms, 1),
            "looking_at_road": self.looking_at_road,
            "attention_score": round(self.attention_score, 1)
        }


class HeadPoseEstimator:
    """
    Estimates head pose and detects distraction.
    
    Uses the pitch, yaw, roll angles from facial landmarks to determine
    if the driver is looking at the road or is distracted.
    
    Detection Logic:
    - Looking down (pitch > 20°) for > 2s = distracted (phone)
    - Looking left/right (|yaw| > 30°) for > 2s = distracted
    - Brief glances (< 1s) are normal driving behavior
    """
    
    # Thresholds for distraction detection (degrees)
    PITCH_DOWN_THRESHOLD = 20.0      # Looking down at phone
    PITCH_UP_THRESHOLD = -25.0       # Looking up (rearview OK briefly)
    YAW_THRESHOLD = 30.0             # Looking left/right
    ROLL_THRESHOLD = 25.0            # Head tilted (fatigue)
    
    # Timing thresholds (milliseconds)
    DISTRACTION_WARNING_MS = 1000    # 1 second
    DISTRACTION_ALERT_MS = 2000      # 2 seconds
    
    def __init__(self):
        """Initialize the head pose estimator"""
        self._distraction_start: Optional[float] = None
        self._current_distraction: DistractionType = DistractionType.NONE
        self._attention_history: list = []
        self._history_size = 30  # ~2 seconds at 15fps
        
    def detect_distraction(self, head_pose: HeadPose) -> DistractionState:
        """
        Analyze head pose and detect distraction.
        
        Args:
            head_pose: HeadPose object with pitch, yaw, roll angles
            
        Returns:
            DistractionState with detection results
        """
        current_time = time.time()
        
        # Determine if currently looking away
        distraction_type = self._classify_head_pose(head_pose)
        looking_at_road = distraction_type == DistractionType.NONE
        
        # Track distraction duration
        if not looking_at_road:
            if self._distraction_start is None:
                self._distraction_start = current_time
                self._current_distraction = distraction_type
            distraction_duration_ms = (current_time - self._distraction_start) * 1000
        else:
            distraction_duration_ms = 0
            self._distraction_start = None
            self._current_distraction = DistractionType.NONE
        
        # Determine if distracted (looking away too long)
        is_distracted = distraction_duration_ms >= self.DISTRACTION_ALERT_MS
        
        # Calculate attention score
        attention_score = self._calculate_attention_score(looking_at_road)
        
        return DistractionState(
            head_pose=head_pose,
            is_distracted=is_distracted,
            distraction_type=distraction_type if not looking_at_road else DistractionType.NONE,
            distraction_duration_ms=distraction_duration_ms,
            looking_at_road=looking_at_road,
            attention_score=attention_score
        )
    
    def _classify_head_pose(self, head_pose: HeadPose) -> DistractionType:
        """Classify the head pose into a distraction type"""
        # Check pitch (up/down)
        if head_pose.pitch > self.PITCH_DOWN_THRESHOLD:
            return DistractionType.LOOKING_DOWN
        if head_pose.pitch < self.PITCH_UP_THRESHOLD:
            return DistractionType.LOOKING_UP
        
        # Check yaw (left/right)
        if head_pose.yaw < -self.YAW_THRESHOLD:
            return DistractionType.LOOKING_LEFT
        if head_pose.yaw > self.YAW_THRESHOLD:
            return DistractionType.LOOKING_RIGHT
        
        # Check roll (head tilt - can indicate fatigue)
        if abs(head_pose.roll) > self.ROLL_THRESHOLD:
            return DistractionType.HEAD_TILTED
        
        return DistractionType.NONE
    
    def _calculate_attention_score(self, looking_at_road: bool) -> float:
        """
        Calculate attention score based on recent history.
        
        Score is 0-100 where 100 = always looking at road.
        """
        # Add current state to history
        self._attention_history.append(1.0 if looking_at_road else 0.0)
        
        # Keep only recent history
        if len(self._attention_history) > self._history_size:
            self._attention_history.pop(0)
        
        # Calculate score as percentage of time looking at road
        if len(self._attention_history) == 0:
            return 100.0
        
        return (sum(self._attention_history) / len(self._attention_history)) * 100
    
    def reset(self):
        """Reset all tracking state"""
        self._distraction_start = None
        self._current_distraction = DistractionType.NONE
        self._attention_history = []
    
    def get_thresholds(self) -> Dict:
        """Get current threshold settings"""
        return {
            "pitch_down": self.PITCH_DOWN_THRESHOLD,
            "pitch_up": self.PITCH_UP_THRESHOLD,
            "yaw": self.YAW_THRESHOLD,
            "roll": self.ROLL_THRESHOLD,
            "warning_ms": self.DISTRACTION_WARNING_MS,
            "alert_ms": self.DISTRACTION_ALERT_MS
        }


# Singleton instance
_head_pose_estimator: Optional[HeadPoseEstimator] = None


def get_head_pose_estimator() -> HeadPoseEstimator:
    """Get or create the singleton HeadPoseEstimator instance"""
    global _head_pose_estimator
    if _head_pose_estimator is None:
        _head_pose_estimator = HeadPoseEstimator()
    return _head_pose_estimator
