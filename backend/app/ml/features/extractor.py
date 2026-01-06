"""
AutoPulse Feature Extractor
Extracts ML-ready features from raw telemetry data
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from app.ml.config import THRESHOLDS, WINDOW_SIZES, MIN_REQUIREMENTS


@dataclass
class TelemetryPoint:
    """Single telemetry reading"""
    timestamp: datetime
    speed_kmh: float
    rpm: int
    gear: int
    throttle_position: float
    engine_temp: float
    oil_temp: float
    oil_pressure: float
    fuel_level: float
    acceleration_g: float = 0.0
    is_harsh_braking: bool = False
    is_harsh_acceleration: bool = False
    is_speeding: bool = False
    is_idling: bool = False
    driving_mode: str = "city"


@dataclass
class TripFeatures:
    """Extracted features for a trip"""
    # Trip metadata
    trip_id: str
    vehicle_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    distance_km: float
    
    # Event counts
    harsh_brake_count: int = 0
    harsh_accel_count: int = 0
    severe_brake_count: int = 0
    severe_accel_count: int = 0
    
    # Time-based metrics (seconds)
    speeding_seconds: int = 0
    dangerous_speed_seconds: int = 0
    over_rpm_seconds: int = 0
    idle_seconds: int = 0
    moving_seconds: int = 0
    high_throttle_seconds: int = 0
    
    # Percentages
    speeding_percentage: float = 0.0
    idle_percentage: float = 0.0
    over_rpm_percentage: float = 0.0
    high_throttle_percentage: float = 0.0
    
    # Speed statistics
    avg_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    speed_std_dev: float = 0.0
    speed_variance: float = 0.0
    
    # Acceleration statistics
    avg_acceleration_g: float = 0.0
    max_acceleration_g: float = 0.0
    max_deceleration_g: float = 0.0
    acceleration_std_dev: float = 0.0
    
    # RPM statistics
    avg_rpm: int = 0
    max_rpm: int = 0
    rpm_std_dev: float = 0.0
    efficient_rpm_percentage: float = 0.0
    
    # Throttle statistics
    avg_throttle: float = 0.0
    throttle_std_dev: float = 0.0
    
    # Engine health
    max_engine_temp: float = 0.0
    avg_engine_temp: float = 0.0
    engine_stress_avg: float = 0.0
    
    # Fuel
    fuel_consumed_percent: float = 0.0
    fuel_efficiency_score: float = 0.0
    
    # Mode breakdown (percentage)
    mode_city_percent: float = 0.0
    mode_highway_percent: float = 0.0
    mode_sport_percent: float = 0.0
    
    # Derived ratios
    harsh_events_per_hour: float = 0.0
    harsh_brake_rate: float = 0.0
    harsh_accel_rate: float = 0.0


class FeatureExtractor:
    """
    Extracts features from raw telemetry data for ML models
    """
    
    def __init__(self):
        self.thresholds = THRESHOLDS
    
    def extract_from_readings(
        self,
        readings: List[Dict],
        trip_id: str = "",
        vehicle_id: str = ""
    ) -> Optional[TripFeatures]:
        """
        Extract features from a list of telemetry readings
        
        Args:
            readings: List of telemetry dictionaries
            trip_id: Trip identifier
            vehicle_id: Vehicle identifier
            
        Returns:
            TripFeatures object or None if insufficient data
        """
        if not readings or len(readings) < MIN_REQUIREMENTS["min_readings_for_score"]:
            return None
        
        # Convert to numpy arrays for efficient computation
        speeds = np.array([r.get('speed_kmh', 0) for r in readings])
        rpms = np.array([r.get('rpm', 0) for r in readings])
        throttles = np.array([r.get('throttle_position', 0) for r in readings])
        engine_temps = np.array([r.get('engine_temp', 0) for r in readings])
        accelerations = np.array([r.get('acceleration_g', 0) for r in readings])
        
        # Boolean arrays
        is_harsh_braking = np.array([r.get('is_harsh_braking', False) for r in readings])
        is_harsh_accel = np.array([r.get('is_harsh_acceleration', False) for r in readings])
        is_speeding = np.array([r.get('is_speeding', False) for r in readings])
        is_idling = np.array([r.get('is_idling', False) for r in readings])
        
        # Driving modes
        modes = [r.get('driving_mode', 'city') for r in readings]
        
        # Time calculations
        timestamps = [r.get('timestamp') or r.get('time') for r in readings]
        if timestamps[0] and timestamps[-1]:
            if isinstance(timestamps[0], str):
                start_time = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            else:
                start_time = timestamps[0]
                end_time = timestamps[-1]
            duration_seconds = int((end_time - start_time).total_seconds())
        else:
            start_time = datetime.now()
            end_time = datetime.now()
            duration_seconds = len(readings)  # Assume 1 reading per second
        
        if duration_seconds < MIN_REQUIREMENTS["min_trip_duration_seconds"]:
            return None
        
        # Calculate all features
        features = TripFeatures(
            trip_id=trip_id,
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            distance_km=self._calculate_distance(speeds, duration_seconds),
        )
        
        # Event counts
        features.harsh_brake_count = self._count_events(is_harsh_braking)
        features.harsh_accel_count = self._count_events(is_harsh_accel)
        features.severe_brake_count = self._count_severe_events(
            accelerations, self.thresholds["severe_brake_g"], is_negative=True
        )
        features.severe_accel_count = self._count_severe_events(
            accelerations, self.thresholds["severe_accel_g"], is_negative=False
        )
        
        # Time-based metrics
        features.speeding_seconds = int(np.sum(is_speeding))
        features.dangerous_speed_seconds = int(np.sum(speeds > self.thresholds["dangerous_speed"]))
        features.over_rpm_seconds = int(np.sum(rpms > self.thresholds["over_rpm"]))
        features.idle_seconds = int(np.sum(is_idling))
        features.moving_seconds = int(np.sum(speeds >= self.thresholds["idle_speed"]))
        features.high_throttle_seconds = int(np.sum(throttles > self.thresholds["high_throttle"]))
        
        # Percentages
        total_readings = len(readings)
        features.speeding_percentage = (features.speeding_seconds / total_readings) * 100
        features.idle_percentage = (features.idle_seconds / total_readings) * 100
        features.over_rpm_percentage = (features.over_rpm_seconds / total_readings) * 100
        features.high_throttle_percentage = (features.high_throttle_seconds / total_readings) * 100
        
        # Speed statistics
        moving_speeds = speeds[speeds >= self.thresholds["idle_speed"]]
        if len(moving_speeds) > 0:
            features.avg_speed_kmh = float(np.mean(moving_speeds))
            features.max_speed_kmh = float(np.max(speeds))
            features.speed_std_dev = float(np.std(moving_speeds))
            features.speed_variance = float(np.var(moving_speeds))
        
        # Acceleration statistics
        non_zero_accel = accelerations[accelerations != 0]
        if len(non_zero_accel) > 0:
            features.avg_acceleration_g = float(np.mean(np.abs(non_zero_accel)))
            features.max_acceleration_g = float(np.max(accelerations))
            features.max_deceleration_g = float(abs(np.min(accelerations)))
            features.acceleration_std_dev = float(np.std(accelerations))
        
        # RPM statistics
        moving_rpms = rpms[speeds >= self.thresholds["idle_speed"]]
        if len(moving_rpms) > 0:
            features.avg_rpm = int(np.mean(moving_rpms))
            features.max_rpm = int(np.max(rpms))
            features.rpm_std_dev = float(np.std(moving_rpms))
            
            # Efficient RPM percentage
            efficient_mask = (moving_rpms >= self.thresholds["efficient_rpm_min"]) & \
                           (moving_rpms <= self.thresholds["efficient_rpm_max"])
            features.efficient_rpm_percentage = (np.sum(efficient_mask) / len(moving_rpms)) * 100
        
        # Throttle statistics
        features.avg_throttle = float(np.mean(throttles))
        features.throttle_std_dev = float(np.std(throttles))
        
        # Engine health
        features.max_engine_temp = float(np.max(engine_temps))
        features.avg_engine_temp = float(np.mean(engine_temps))
        
        # Engine stress (from readings if available)
        stress_scores = [r.get('engine_stress_score', 0) for r in readings]
        features.engine_stress_avg = float(np.mean(stress_scores)) if stress_scores else 0
        
        # Fuel
        fuel_levels = [r.get('fuel_level', 0) for r in readings]
        if fuel_levels:
            features.fuel_consumed_percent = max(0, fuel_levels[0] - fuel_levels[-1])
        
        # Mode breakdown
        mode_counts = {'city': 0, 'highway': 0, 'sport': 0, 'parked': 0, 'reverse': 0}
        for mode in modes:
            if mode in mode_counts:
                mode_counts[mode] += 1
        
        moving_total = mode_counts['city'] + mode_counts['highway'] + mode_counts['sport']
        if moving_total > 0:
            features.mode_city_percent = (mode_counts['city'] / moving_total) * 100
            features.mode_highway_percent = (mode_counts['highway'] / moving_total) * 100
            features.mode_sport_percent = (mode_counts['sport'] / moving_total) * 100
        
        # Derived rates (per hour)
        hours = duration_seconds / 3600
        if hours > 0:
            features.harsh_events_per_hour = (features.harsh_brake_count + features.harsh_accel_count) / hours
            features.harsh_brake_rate = features.harsh_brake_count / hours
            features.harsh_accel_rate = features.harsh_accel_count / hours
        
        return features
    
    def _calculate_distance(self, speeds: np.ndarray, duration_seconds: int) -> float:
        """Estimate distance from speed readings"""
        if len(speeds) == 0 or duration_seconds == 0:
            return 0.0
        
        # Average speed in km/h, convert to km for duration
        avg_speed = np.mean(speeds)
        hours = duration_seconds / 3600
        return avg_speed * hours
    
    def _count_events(self, bool_array: np.ndarray) -> int:
        """Count distinct events (transitions from False to True)"""
        if len(bool_array) == 0:
            return 0
        
        # Find transitions from False to True
        transitions = np.diff(bool_array.astype(int))
        return int(np.sum(transitions == 1))
    
    def _count_severe_events(
        self,
        accelerations: np.ndarray,
        threshold: float,
        is_negative: bool
    ) -> int:
        """Count severe acceleration/braking events"""
        if is_negative:
            severe = accelerations < threshold
        else:
            severe = accelerations > threshold
        
        return self._count_events(severe)
    
    def features_to_dict(self, features: TripFeatures) -> Dict:
        """Convert TripFeatures to dictionary for JSON serialization"""
        return {
            "trip_id": features.trip_id,
            "vehicle_id": features.vehicle_id,
            "start_time": features.start_time.isoformat() if features.start_time else None,
            "end_time": features.end_time.isoformat() if features.end_time else None,
            "duration_seconds": features.duration_seconds,
            "distance_km": round(features.distance_km, 2),
            
            "events": {
                "harsh_brake_count": features.harsh_brake_count,
                "harsh_accel_count": features.harsh_accel_count,
                "severe_brake_count": features.severe_brake_count,
                "severe_accel_count": features.severe_accel_count,
            },
            
            "time_metrics": {
                "speeding_seconds": features.speeding_seconds,
                "idle_seconds": features.idle_seconds,
                "over_rpm_seconds": features.over_rpm_seconds,
                "high_throttle_seconds": features.high_throttle_seconds,
            },
            
            "percentages": {
                "speeding": round(features.speeding_percentage, 1),
                "idle": round(features.idle_percentage, 1),
                "over_rpm": round(features.over_rpm_percentage, 1),
                "high_throttle": round(features.high_throttle_percentage, 1),
                "efficient_rpm": round(features.efficient_rpm_percentage, 1),
            },
            
            "speed_stats": {
                "avg": round(features.avg_speed_kmh, 1),
                "max": round(features.max_speed_kmh, 1),
                "std_dev": round(features.speed_std_dev, 2),
            },
            
            "acceleration_stats": {
                "avg_g": round(features.avg_acceleration_g, 3),
                "max_g": round(features.max_acceleration_g, 3),
                "max_decel_g": round(features.max_deceleration_g, 3),
            },
            
            "rpm_stats": {
                "avg": features.avg_rpm,
                "max": features.max_rpm,
            },
            
            "rates": {
                "harsh_events_per_hour": round(features.harsh_events_per_hour, 1),
                "harsh_brake_per_hour": round(features.harsh_brake_rate, 1),
                "harsh_accel_per_hour": round(features.harsh_accel_rate, 1),
            },
            
            "mode_breakdown": {
                "city": round(features.mode_city_percent, 1),
                "highway": round(features.mode_highway_percent, 1),
                "sport": round(features.mode_sport_percent, 1),
            },
        }


# Create singleton instance
extractor = FeatureExtractor()
