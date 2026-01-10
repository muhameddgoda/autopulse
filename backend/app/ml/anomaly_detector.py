"""
AutoPulse Anomaly Detection Service
===================================
Detects anomalies in vehicle telemetry data using Isolation Forest.

Isolation Forest is ideal for automotive telemetry because:
1. It doesn't require labeled training data (unsupervised)
2. Works well with multi-dimensional sensor data
3. Fast inference for real-time detection
4. Handles varying sensor distributions

Anomaly Types Detected:
- Sensor malfunction (sudden spikes/drops)
- Unusual driving patterns (RPM/speed mismatch)
- Component degradation (gradual drift)
- Environmental factors (extreme temps)

Usage:
    detector = AnomalyDetector()
    
    # Train on normal data
    detector.fit(telemetry_df)
    
    # Detect anomalies
    result = detector.detect(new_reading)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import pickle
from datetime import datetime, timezone

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    StandardScaler = None

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of detected anomalies"""
    NONE = "none"
    SENSOR_SPIKE = "sensor_spike"
    SENSOR_DROP = "sensor_drop"
    PATTERN_ANOMALY = "pattern_anomaly"
    DEGRADATION = "degradation"
    OUT_OF_RANGE = "out_of_range"


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single reading"""
    is_anomaly: bool
    anomaly_score: float  # -1 to 0 (more negative = more anomalous)
    anomaly_probability: float  # 0 to 1 (higher = more anomalous)
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    anomalous_features: List[str]  # Which features are anomalous
    details: Dict
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        # Convert numpy types to native Python types for JSON serialization
        return {
            "is_anomaly": bool(self.is_anomaly),
            "anomaly_score": float(round(self.anomaly_score, 4)),
            "anomaly_probability": float(round(self.anomaly_probability, 3)),
            "severity": self.severity.value,
            "anomaly_type": self.anomaly_type.value,
            "anomalous_features": list(self.anomalous_features),
            "details": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else 
                          bool(v) if isinstance(v, np.bool_) else v) 
                       for k, v in self.details.items()},
            "timestamp": self.timestamp.isoformat()
        }


class TelemetryAnomalyDetector:
    """
    Anomaly detection for vehicle telemetry using Isolation Forest.
    
    Features analyzed:
    - speed_kmh, rpm, engine_temp_c, coolant_temp_c
    - fuel_level_pct, battery_voltage, oil_pressure_bar
    - throttle_position_pct, engine_load_pct
    - tire_pressure (FL, FR, RL, RR)
    
    Also computes derived features:
    - speed_rpm_ratio (gear indicator)
    - temp_delta (engine - coolant)
    - tire_variance (consistency)
    """
    
    # Default feature columns for telemetry
    FEATURE_COLUMNS = [
        'speed_kmh', 'rpm', 'engine_temp_c', 'coolant_temp_c',
        'fuel_level_pct', 'battery_voltage', 'oil_pressure_bar',
        'throttle_position_pct', 'engine_load_pct',
        'tire_pressure_fl', 'tire_pressure_fr',
        'tire_pressure_rl', 'tire_pressure_rr'
    ]
    
    # Expected ranges for rule-based checks
    EXPECTED_RANGES = {
        'speed_kmh': (0, 260),
        'rpm': (0, 8000),
        'engine_temp_c': (60, 120),
        'coolant_temp_c': (60, 110),
        'fuel_level_pct': (0, 100),
        'battery_voltage': (11.5, 14.8),
        'oil_pressure_bar': (1.0, 6.0),
        'throttle_position_pct': (0, 100),
        'engine_load_pct': (0, 100),
        'tire_pressure_fl': (1.8, 3.5),
        'tire_pressure_fr': (1.8, 3.5),
        'tire_pressure_rl': (1.8, 3.5),
        'tire_pressure_rr': (1.8, 3.5)
    }
    
    def __init__(
        self,
        contamination: float=0.05,  # Expected % of anomalies
        n_estimators: int=100,
        random_state: int=42
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers (0.01-0.1)
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Anomaly detection disabled.")
            self._enabled = False
            return
        
        self._enabled = True
        self._contamination = contamination
        
        # Initialize Isolation Forest
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Scaler for feature normalization
        self._scaler = StandardScaler()
        
        # Tracking
        self._is_fitted = False
        self._feature_columns = self.FEATURE_COLUMNS.copy()
        self._anomaly_count = 0
        self._total_processed = 0
        
        # Rolling statistics for drift detection
        self._recent_values: Dict[str, List[float]] = {col: [] for col in self.FEATURE_COLUMNS}
        self._history_size = 100
        
        logger.info(f"TelemetryAnomalyDetector initialized (contamination={contamination})")
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed features for better anomaly detection"""
        df = df.copy()
        
        # Speed-RPM ratio (gear indicator)
        df['speed_rpm_ratio'] = np.where(
            df['rpm'] > 0,
            df['speed_kmh'] / (df['rpm'] / 1000),
            0
        )
        
        # Temperature delta
        if 'engine_temp_c' in df.columns and 'coolant_temp_c' in df.columns:
            df['temp_delta'] = df['engine_temp_c'] - df['coolant_temp_c']
        
        # Tire pressure variance
        tire_cols = ['tire_pressure_fl', 'tire_pressure_fr',
                     'tire_pressure_rl', 'tire_pressure_rr']
        if all(col in df.columns for col in tire_cols):
            df['tire_variance'] = df[tire_cols].var(axis=1)
        
        # Throttle-load correlation check
        df['throttle_load_ratio'] = np.where(
            df['throttle_position_pct'] > 5,
            df['engine_load_pct'] / df['throttle_position_pct'],
            1.0
        )
        
        return df
    
    def fit(self, telemetry_data: pd.DataFrame) -> 'TelemetryAnomalyDetector':
        """
        Train the anomaly detector on historical telemetry data.
        
        Args:
            telemetry_data: DataFrame with telemetry readings
            
        Returns:
            self for method chaining
        """
        if not self._enabled:
            logger.warning("Anomaly detection disabled, skipping fit")
            return self
        
        logger.info(f"Fitting anomaly detector on {len(telemetry_data)} samples")
        
        # Select available features
        available_cols = [col for col in self.FEATURE_COLUMNS 
                         if col in telemetry_data.columns]
        self._feature_columns = available_cols
        
        if len(available_cols) < 3:
            logger.error(f"Insufficient features: {available_cols}")
            return self
        
        # Prepare data
        df = telemetry_data[available_cols].copy()
        df = df.dropna()
        
        if len(df) < 10:
            logger.warning("Insufficient data for training")
            return self
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Get final feature columns
        final_cols = [c for c in df.columns if c in available_cols or 
                      c in ['speed_rpm_ratio', 'temp_delta', 'tire_variance', 'throttle_load_ratio']]
        
        # Scale features
        X = df[final_cols].values
        X_scaled = self._scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self._model.fit(X_scaled)
        self._is_fitted = True
        self._feature_columns = final_cols
        
        logger.info(f"Anomaly detector fitted with features: {final_cols}")
        return self
    
    def detect(self, reading: Dict) -> AnomalyResult:
        """
        Detect anomalies in a single telemetry reading.
        
        Args:
            reading: Dict with telemetry values
            
        Returns:
            AnomalyResult with detection details
        """
        timestamp = datetime.now(timezone.utc)
        self._total_processed += 1
        
        # Default result for disabled detector
        if not self._enabled or not self._is_fitted:
            return self._rule_based_detection(reading, timestamp)
        
        # Prepare features - only base features, derived ones will be added
        base_features = [col for col in self.FEATURE_COLUMNS]
        features = {}
        for col in base_features:
            features[col] = reading.get(col, 0)
        
        # Create DataFrame for processing
        df = pd.DataFrame([features])
        
        # Handle missing values - use 0 as default
        if df.isna().any().any():
            df = df.fillna(0)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Align columns to match trained model
        try:
            # Ensure all required columns exist
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            X = df[self._feature_columns].values
            X_scaled = self._scaler.transform(X)
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return self._rule_based_detection(reading, timestamp)
        
        # Get Isolation Forest prediction
        prediction = self._model.predict(X_scaled)[0]
        score = self._model.decision_function(X_scaled)[0]
        
        # -1 = anomaly, 1 = normal in sklearn
        is_anomaly = prediction == -1
        
        # Convert score to probability (roughly)
        # Score ranges from about -0.5 (very anomalous) to 0.5 (very normal)
        anomaly_probability = max(0, min(1, 0.5 - score))
        
        # Identify anomalous features
        anomalous_features = self._identify_anomalous_features(reading)
        
        # Determine severity
        severity = self._calculate_severity(score, anomalous_features)
        
        # Determine anomaly type
        anomaly_type = self._classify_anomaly(reading, anomalous_features)
        
        # Update tracking
        if is_anomaly:
            self._anomaly_count += 1
        
        # Update rolling statistics
        self._update_rolling_stats(reading)
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_probability=anomaly_probability,
            severity=severity,
            anomaly_type=anomaly_type,
            anomalous_features=anomalous_features,
            details={
                "raw_prediction": int(prediction),
                "feature_count": len(self._feature_columns),
                "total_processed": self._total_processed,
                "anomaly_rate": round(self._anomaly_count / self._total_processed, 4)
            },
            timestamp=timestamp
        )
    
    def _rule_based_detection(self, reading: Dict, timestamp: datetime) -> AnomalyResult:
        """Fallback rule-based anomaly detection"""
        anomalous_features = []
        max_severity_score = 0
        
        for feature, (min_val, max_val) in self.EXPECTED_RANGES.items():
            value = reading.get(feature)
            if value is None:
                continue
            
            if value < min_val or value > max_val:
                anomalous_features.append(feature)
                # Calculate how far out of range
                if value < min_val:
                    deviation = (min_val - value) / (min_val + 0.001)
                else:
                    deviation = (value - max_val) / (max_val + 0.001)
                max_severity_score = max(max_severity_score, deviation)
        
        is_anomaly = len(anomalous_features) > 0
        
        if max_severity_score > 0.5:
            severity = AnomalySeverity.CRITICAL
        elif max_severity_score > 0.3:
            severity = AnomalySeverity.HIGH
        elif max_severity_score > 0.1:
            severity = AnomalySeverity.MEDIUM
        elif is_anomaly:
            severity = AnomalySeverity.LOW
        else:
            severity = AnomalySeverity.NONE
        
        anomaly_type = AnomalyType.OUT_OF_RANGE if is_anomaly else AnomalyType.NONE
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=-max_severity_score if is_anomaly else 0,
            anomaly_probability=min(1.0, max_severity_score) if is_anomaly else 0,
            severity=severity,
            anomaly_type=anomaly_type,
            anomalous_features=anomalous_features,
            details={"detection_method": "rule_based"},
            timestamp=timestamp
        )
    
    def _identify_anomalous_features(self, reading: Dict) -> List[str]:
        """Identify which specific features are anomalous"""
        anomalous = []
        
        for feature, (min_val, max_val) in self.EXPECTED_RANGES.items():
            value = reading.get(feature)
            if value is None:
                continue
            
            # Check against expected range with margin
            margin = (max_val - min_val) * 0.1
            if value < (min_val - margin) or value > (max_val + margin):
                anomalous.append(feature)
            
            # Check against rolling statistics (if available)
            if feature in self._recent_values and len(self._recent_values[feature]) >= 10:
                recent = self._recent_values[feature]
                mean = np.mean(recent)
                std = np.std(recent) + 0.001  # Avoid division by zero
                z_score = abs(value - mean) / std
                if z_score > 3:  # More than 3 standard deviations
                    if feature not in anomalous:
                        anomalous.append(feature)
        
        return anomalous
    
    def _calculate_severity(self, score: float, anomalous_features: List[str]) -> AnomalySeverity:
        """Calculate anomaly severity based on score and features"""
        # Critical sensors that affect safety
        critical_sensors = {'battery_voltage', 'oil_pressure_bar', 'engine_temp_c', 'coolant_temp_c'}
        has_critical = any(f in critical_sensors for f in anomalous_features)
        
        if score < -0.3:
            return AnomalySeverity.CRITICAL if has_critical else AnomalySeverity.HIGH
        elif score < -0.2:
            return AnomalySeverity.HIGH if has_critical else AnomalySeverity.MEDIUM
        elif score < -0.1:
            return AnomalySeverity.MEDIUM if has_critical else AnomalySeverity.LOW
        elif score < 0:
            return AnomalySeverity.LOW
        return AnomalySeverity.NONE
    
    def _classify_anomaly(self, reading: Dict, anomalous_features: List[str]) -> AnomalyType:
        """Classify the type of anomaly detected"""
        if not anomalous_features:
            return AnomalyType.NONE
        
        # Check for sudden spikes
        for feature in anomalous_features:
            if feature in self._recent_values and len(self._recent_values[feature]) >= 5:
                recent = self._recent_values[feature][-5:]
                current = reading.get(feature, 0)
                avg = np.mean(recent)
                
                if current > avg * 1.5:
                    return AnomalyType.SENSOR_SPIKE
                elif current < avg * 0.5:
                    return AnomalyType.SENSOR_DROP
        
        # Check for pattern anomalies (multiple correlated sensors)
        if len(anomalous_features) >= 3:
            return AnomalyType.PATTERN_ANOMALY
        
        return AnomalyType.OUT_OF_RANGE
    
    def _update_rolling_stats(self, reading: Dict):
        """Update rolling statistics for drift detection"""
        for feature in self.FEATURE_COLUMNS:
            value = reading.get(feature)
            if value is not None:
                self._recent_values[feature].append(value)
                if len(self._recent_values[feature]) > self._history_size:
                    self._recent_values[feature].pop(0)
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            "enabled": self._enabled,
            "is_fitted": self._is_fitted,
            "total_processed": self._total_processed,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": round(self._anomaly_count / max(1, self._total_processed), 4),
            "feature_count": len(self._feature_columns),
            "contamination": self._contamination
        }
    
    def save_model(self, path: str):
        """Save the trained model to disk"""
        if not self._is_fitted:
            logger.warning("Model not fitted, nothing to save")
            return
        
        model_data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_columns": self._feature_columns,
            "contamination": self._contamination
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Anomaly model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self._model = model_data["model"]
        self._scaler = model_data["scaler"]
        self._feature_columns = model_data["feature_columns"]
        self._contamination = model_data["contamination"]
        self._is_fitted = True
        
        logger.info(f"Anomaly model loaded from {path}")
    
    def reset_statistics(self):
        """Reset tracking statistics"""
        self._anomaly_count = 0
        self._total_processed = 0
        self._recent_values = {col: [] for col in self.FEATURE_COLUMNS}


# Singleton instance
_anomaly_detector: Optional[TelemetryAnomalyDetector] = None


def get_anomaly_detector() -> TelemetryAnomalyDetector:
    """Get or create the singleton anomaly detector"""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = TelemetryAnomalyDetector()
    return _anomaly_detector


def is_anomaly_detection_available() -> bool:
    """Check if anomaly detection is available"""
    return SKLEARN_AVAILABLE
