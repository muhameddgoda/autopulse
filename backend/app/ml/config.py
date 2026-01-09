"""
AutoPulse ML Configuration
Thresholds, weights, and settings for driver behavior analysis
"""

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================

THRESHOLDS = {
    # Harsh event detection (in G-force)
    "harsh_brake_g":-0.4,  # Deceleration beyond this is harsh
    "harsh_accel_g": 0.35,  # Acceleration beyond this is harsh
    "severe_brake_g":-0.6,  # Severe/emergency braking
    "severe_accel_g": 0.5,  # Very aggressive acceleration
    
    # Speed thresholds (km/h)
    "speeding_urban": 60,  # Urban speed limit
    "speeding_highway": 130,  # Highway speed limit
    "dangerous_speed": 180,  # Dangerous speed
    
    # RPM thresholds
    "over_rpm": 6500,  # High RPM threshold
    "redline_rpm": 7200,  # Redline - engine stress
    "efficient_rpm_min": 1500,  # Efficient driving range
    "efficient_rpm_max": 3500,  # Efficient driving range
    
    # Engine health
    "high_engine_temp": 105,  # °C - warning
    "critical_engine_temp": 115,  # °C - critical
    
    # Throttle
    "aggressive_throttle": 85,  # % - aggressive
    "high_throttle": 70,  # % - high
    
    # Idle
    "idle_speed": 5,  # km/h - below is idle
    "excessive_idle_minutes": 5,  # Minutes of idle = excessive
}

# =============================================================================
# SCORING WEIGHTS
# =============================================================================

# Weight distribution for driver score (must sum to 1.0)
SCORING_WEIGHTS = {
    "harsh_braking": 0.20,  # 20% - Safety critical
    "harsh_acceleration": 0.15,  # 15% - Aggressive behavior
    "speeding": 0.20,  # 20% - Safety critical
    "speed_consistency": 0.10,  # 10% - Smooth driving
    "rpm_efficiency": 0.10,  # 10% - Engine care
    "throttle_smoothness": 0.10,  # 10% - Fuel efficiency
    "idle_time": 0.05,  # 5% - Fuel waste
    "engine_stress": 0.10,  # 10% - Vehicle health
}

# =============================================================================
# SCORING PENALTIES
# =============================================================================

# Points deducted per event/occurrence
PENALTIES = {
    # Per harsh event
    "harsh_brake_event": 3,  # -3 points per harsh brake
    "harsh_accel_event": 2,  # -2 points per harsh accel
    "severe_brake_event": 8,  # -8 points per severe brake
    "severe_accel_event": 5,  # -5 points per severe accel
    
    # Per minute of violation
    "speeding_per_minute": 1,  # -1 point per minute speeding
    "dangerous_speed_per_minute": 3,  # -3 points per minute dangerous
    "over_rpm_per_minute": 0.5,  # -0.5 points per minute over RPM
    "excessive_idle_per_minute": 0.3,  # -0.3 points per minute idle
    
    # Thresholds for percentage-based penalties
    "speeding_percentage_threshold": 10,  # Start penalizing above 10%
    "idle_percentage_threshold": 15,  # Start penalizing above 15%
}

# =============================================================================
# BEHAVIOR CLASSIFICATION
# =============================================================================

# Score ranges for behavior labels
BEHAVIOR_LABELS = {
    "exemplary": (90, 100),  # 90-100: Exemplary driver
    "calm": (75, 89),  # 75-89: Calm, safe driver
    "normal": (60, 74),  # 60-74: Average driver
    "aggressive": (40, 59),  # 40-59: Aggressive tendencies
    "dangerous": (0, 39),  # 0-39: Dangerous driving
}

# Risk levels
RISK_LEVELS = {
    "low": (75, 100),
    "medium": (50, 74),
    "high": (25, 49),
    "critical": (0, 24),
}

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Rolling window sizes for feature calculation
WINDOW_SIZES = {
    "short": 30,  # 30 seconds - immediate behavior
    "medium": 300,  # 5 minutes - trip segment
    "long": 900,  # 15 minutes - extended behavior
}

# Minimum data requirements
MIN_REQUIREMENTS = {
    "min_trip_duration_seconds": 30,  # At least 30 seconds
    "min_readings_for_score": 30,  # At least 30 readings
    "min_moving_time_seconds": 30,  # At least 30 seconds moving
}

# =============================================================================
# ML MODEL SETTINGS
# =============================================================================

ML_SETTINGS = {
    # Feature scaling
    "use_standard_scaler": True,
    
    # Model persistence
    "model_path": "app/ml/trained_models/",
    "data_path": "app/ml/training_data/",
    
    # Training settings
    "test_split": 0.2,
    "random_state": 42,
    
    # Classification labels
    "behavior_classes": ["calm", "normal", "aggressive"],
}

# =============================================================================
# INSIGHTS & RECOMMENDATIONS
# =============================================================================

INSIGHT_TEMPLATES = {
    "harsh_braking": {
        "few": "{count} harsh braking events - good control",
        "moderate": "{count} harsh braking events detected",
        "many": "{count} harsh braking events - needs improvement",
    },
    "harsh_acceleration": {
        "few": "Smooth acceleration patterns",
        "moderate": "{count} rapid acceleration events",
        "many": "{count} aggressive acceleration events - reduce throttle aggression",
    },
    "speeding": {
        "none": "No speeding detected - excellent compliance",
        "low": "Minor speeding ({percentage:.1f}% of trip)",
        "moderate": "Speeding {percentage:.1f}% of trip time",
        "high": "Excessive speeding ({percentage:.1f}%) - significant safety risk",
    },
    "idle": {
        "low": "Minimal idle time - good efficiency",
        "moderate": "{minutes:.1f} minutes idle time",
        "high": "Excessive idling ({minutes:.1f} min) - wasting fuel",
    },
}

RECOMMENDATION_TEMPLATES = {
    "harsh_braking": "Maintain greater following distance to avoid sudden stops",
    "harsh_acceleration": "Accelerate gradually for better fuel economy and safety",
    "speeding": "Reduce speed, especially in urban areas",
    "dangerous_speed": "Significantly reduce speed - current driving is dangerous",
    "over_rpm": "Shift to higher gear earlier for better fuel efficiency",
    "high_idle": "Turn off engine during extended stops to save fuel",
    "aggressive_throttle": "Ease off the throttle for smoother, safer driving",
    "engine_stress": "Reduce aggressive driving to prevent engine wear",
}
