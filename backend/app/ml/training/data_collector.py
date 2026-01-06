"""
AutoPulse ML Data Collector
Collects and prepares training data from trips
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

from app.ml.features.extractor import FeatureExtractor, TripFeatures


@dataclass
class TrainingExample:
    """Single training example with features and labels"""
    trip_id: str
    vehicle_id: str
    timestamp: str
    
    # Features (from TripFeatures)
    duration_seconds: int
    distance_km: float
    
    # Event features
    harsh_brake_count: int
    harsh_accel_count: int
    harsh_brake_rate: float
    harsh_accel_rate: float
    
    # Percentage features
    speeding_percentage: float
    idle_percentage: float
    over_rpm_percentage: float
    high_throttle_percentage: float
    
    # Speed features
    avg_speed_kmh: float
    max_speed_kmh: float
    speed_std_dev: float
    
    # Acceleration features
    avg_acceleration_g: float
    max_acceleration_g: float
    max_deceleration_g: float
    
    # RPM features
    avg_rpm: int
    max_rpm: int
    efficient_rpm_percentage: float
    
    # Other
    avg_throttle: float
    engine_stress_avg: float
    mode_sport_percent: float
    
    # Labels (for supervised learning)
    behavior_label: str = ""      # calm, normal, aggressive
    risk_label: str = ""          # low, medium, high
    rule_based_score: float = 0   # Score from rule-based system


class DataCollector:
    """
    Collects training data from analyzed trips
    """
    
    def __init__(self, data_dir: str = "app/ml/training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
    
    def features_to_training_example(
        self,
        features: TripFeatures,
        behavior_label: str = "",
        risk_label: str = "",
        rule_score: float = 0
    ) -> TrainingExample:
        """Convert TripFeatures to TrainingExample"""
        return TrainingExample(
            trip_id=features.trip_id,
            vehicle_id=features.vehicle_id,
            timestamp=features.start_time.isoformat() if features.start_time else "",
            
            duration_seconds=features.duration_seconds,
            distance_km=features.distance_km,
            
            harsh_brake_count=features.harsh_brake_count,
            harsh_accel_count=features.harsh_accel_count,
            harsh_brake_rate=features.harsh_brake_rate,
            harsh_accel_rate=features.harsh_accel_rate,
            
            speeding_percentage=features.speeding_percentage,
            idle_percentage=features.idle_percentage,
            over_rpm_percentage=features.over_rpm_percentage,
            high_throttle_percentage=features.high_throttle_percentage,
            
            avg_speed_kmh=features.avg_speed_kmh,
            max_speed_kmh=features.max_speed_kmh,
            speed_std_dev=features.speed_std_dev,
            
            avg_acceleration_g=features.avg_acceleration_g,
            max_acceleration_g=features.max_acceleration_g,
            max_deceleration_g=features.max_deceleration_g,
            
            avg_rpm=features.avg_rpm,
            max_rpm=features.max_rpm,
            efficient_rpm_percentage=features.efficient_rpm_percentage,
            
            avg_throttle=features.avg_throttle,
            engine_stress_avg=features.engine_stress_avg,
            mode_sport_percent=features.mode_sport_percent,
            
            behavior_label=behavior_label,
            risk_label=risk_label,
            rule_based_score=rule_score
        )
    
    def save_training_data(
        self,
        examples: List[TrainingExample],
        filename: str = "training_data.csv"
    ):
        """Save training examples to CSV"""
        filepath = self.data_dir / filename
        
        if not examples:
            print("No examples to save")
            return
        
        # Get field names from dataclass
        fieldnames = list(asdict(examples[0]).keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for example in examples:
                writer.writerow(asdict(example))
        
        print(f"✅ Saved {len(examples)} examples to {filepath}")
    
    def load_training_data(
        self,
        filename: str = "training_data.csv"
    ) -> List[TrainingExample]:
        """Load training examples from CSV"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return []
        
        examples = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert types
                example = TrainingExample(
                    trip_id=row['trip_id'],
                    vehicle_id=row['vehicle_id'],
                    timestamp=row['timestamp'],
                    duration_seconds=int(row['duration_seconds']),
                    distance_km=float(row['distance_km']),
                    harsh_brake_count=int(row['harsh_brake_count']),
                    harsh_accel_count=int(row['harsh_accel_count']),
                    harsh_brake_rate=float(row['harsh_brake_rate']),
                    harsh_accel_rate=float(row['harsh_accel_rate']),
                    speeding_percentage=float(row['speeding_percentage']),
                    idle_percentage=float(row['idle_percentage']),
                    over_rpm_percentage=float(row['over_rpm_percentage']),
                    high_throttle_percentage=float(row['high_throttle_percentage']),
                    avg_speed_kmh=float(row['avg_speed_kmh']),
                    max_speed_kmh=float(row['max_speed_kmh']),
                    speed_std_dev=float(row['speed_std_dev']),
                    avg_acceleration_g=float(row['avg_acceleration_g']),
                    max_acceleration_g=float(row['max_acceleration_g']),
                    max_deceleration_g=float(row['max_deceleration_g']),
                    avg_rpm=int(row['avg_rpm']),
                    max_rpm=int(row['max_rpm']),
                    efficient_rpm_percentage=float(row['efficient_rpm_percentage']),
                    avg_throttle=float(row['avg_throttle']),
                    engine_stress_avg=float(row['engine_stress_avg']),
                    mode_sport_percent=float(row['mode_sport_percent']),
                    behavior_label=row['behavior_label'],
                    risk_label=row['risk_label'],
                    rule_based_score=float(row['rule_based_score'])
                )
                examples.append(example)
        
        print(f"✅ Loaded {len(examples)} examples from {filepath}")
        return examples
    
    def prepare_features_matrix(
        self,
        examples: List[TrainingExample]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert training examples to feature matrix for ML
        
        Returns:
            X: numpy array of features (n_samples, n_features)
            feature_names: list of feature names
        """
        feature_names = [
            'duration_seconds',
            'distance_km',
            'harsh_brake_count',
            'harsh_accel_count',
            'harsh_brake_rate',
            'harsh_accel_rate',
            'speeding_percentage',
            'idle_percentage',
            'over_rpm_percentage',
            'high_throttle_percentage',
            'avg_speed_kmh',
            'max_speed_kmh',
            'speed_std_dev',
            'avg_acceleration_g',
            'max_acceleration_g',
            'max_deceleration_g',
            'avg_rpm',
            'max_rpm',
            'efficient_rpm_percentage',
            'avg_throttle',
            'engine_stress_avg',
            'mode_sport_percent',
        ]
        
        X = np.array([
            [
                ex.duration_seconds,
                ex.distance_km,
                ex.harsh_brake_count,
                ex.harsh_accel_count,
                ex.harsh_brake_rate,
                ex.harsh_accel_rate,
                ex.speeding_percentage,
                ex.idle_percentage,
                ex.over_rpm_percentage,
                ex.high_throttle_percentage,
                ex.avg_speed_kmh,
                ex.max_speed_kmh,
                ex.speed_std_dev,
                ex.avg_acceleration_g,
                ex.max_acceleration_g,
                ex.max_deceleration_g,
                ex.avg_rpm,
                ex.max_rpm,
                ex.efficient_rpm_percentage,
                ex.avg_throttle,
                ex.engine_stress_avg,
                ex.mode_sport_percent,
            ]
            for ex in examples
        ])
        
        return X, feature_names
    
    def get_behavior_labels(
        self,
        examples: List[TrainingExample]
    ) -> np.ndarray:
        """Get behavior labels as numpy array"""
        label_map = {'calm': 0, 'normal': 1, 'aggressive': 2}
        return np.array([
            label_map.get(ex.behavior_label, 1)  # default to normal
            for ex in examples
        ])
    
    def get_risk_labels(
        self,
        examples: List[TrainingExample]
    ) -> np.ndarray:
        """Get risk labels as numpy array"""
        label_map = {'low': 0, 'medium': 1, 'high': 2}
        return np.array([
            label_map.get(ex.risk_label, 1)
            for ex in examples
        ])
    
    def get_scores(
        self,
        examples: List[TrainingExample]
    ) -> np.ndarray:
        """Get rule-based scores for regression"""
        return np.array([ex.rule_based_score for ex in examples])


# Create singleton
data_collector = DataCollector()
