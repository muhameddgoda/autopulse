"""
AutoPulse Anomaly Detector
Detects unusual driving patterns using unsupervised learning

Algorithms:
- Isolation Forest (default, fast, handles high-dimensional data)
- One-Class SVM (alternative, good for small datasets)
- Local Outlier Factor (good for density-based anomalies)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.ml.training.data_collector import DataCollector, TrainingExample


class AnomalyDetector:
    """
    Detects anomalous driving patterns
    
    Use cases:
    - Unusual acceleration/braking patterns
    - Abnormal speed variations
    - Potential vehicle issues (indicated by unusual patterns)
    - Driver impairment detection
    """
    
    # Compute default model directory relative to this file
    _DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "trained_models"
    
    def __init__(self, model_dir: str=None):
        self.model_dir = Path(model_dir) if model_dir else self._DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Anomaly thresholds
        self.contamination = 0.1  # Expected fraction of anomalies
        
        # Training metadata
        self.metadata = {
            "trained_at": None,
            "n_samples": 0,
            "algorithm": "",
            "contamination": self.contamination,
            "feature_stats": {}
        }
    
    def train(
        self,
        examples: List[TrainingExample],
        algorithm: str="isolation_forest",
        contamination: float=0.1
    ) -> Dict:
        """
        Train the anomaly detector
        
        Args:
            examples: Training examples (normal driving patterns)
            algorithm: 'isolation_forest', 'one_class_svm', or 'lof'
            contamination: Expected fraction of anomalies
            
        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed"}
        
        if len(examples) < 20:
            return {"error": f"Need at least 20 examples, got {len(examples)}"}
        
        self.contamination = contamination
        
        # Prepare data
        collector = DataCollector()
        X, self.feature_names = collector.prepare_features_matrix(examples)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate feature statistics (for interpretation)
        feature_stats = {}
        for i, name in enumerate(self.feature_names):
            feature_stats[name] = {
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i])),
                "min": float(np.min(X[:, i])),
                "max": float(np.max(X[:, i])),
                "median": float(np.median(X[:, i]))
            }
        
        # Select algorithm
        print(f"🚀 Training {algorithm} anomaly detector...")
        
        if algorithm == "one_class_svm":
            self.model = OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=contamination
            )
        elif algorithm == "lof":
            # LOF is a bit different - it's fitted at prediction time
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True  # Enable predict() method
            )
        else:
            # Default: Isolation Forest
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        
        # Train
        self.model.fit(X_scaled)
        
        # Get anomaly scores for training data
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_scaled)
        else:
            scores = self.model.score_samples(X_scaled)
        
        # Predict on training data to get baseline
        predictions = self.model.predict(X_scaled)
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(examples)
        
        # Update metadata
        self.metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(examples),
            "algorithm": algorithm,
            "contamination": contamination,
            "actual_anomaly_rate": float(anomaly_rate),
            "score_threshold": float(np.percentile(scores, contamination * 100)),
            "feature_stats": feature_stats,
            "feature_names": self.feature_names
        }
        
        self.is_trained = True
        
        results = {
            "success": True,
            "n_samples": len(examples),
            "algorithm": algorithm,
            "contamination": contamination,
            "anomalies_in_training": int(n_anomalies),
            "anomaly_rate": float(anomaly_rate),
            "score_stats": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            },
            "metadata": self.metadata
        }
        
        print(f"✅ Training complete!")
        print(f"   Detected {n_anomalies} anomalies in training data ({anomaly_rate:.1%})")
        
        return results
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect if a trip is anomalous
        
        Args:
            features: Feature array (1, n_features)
            
        Returns:
            (is_anomaly, anomaly_score, details)
        """
        if not self.is_trained or self.model is None:
            return False, 0.0, {"error": "Model not trained"}
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        is_anomaly = prediction == -1
        
        # Get anomaly score
        if hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(features_scaled)[0]
        else:
            score = self.model.score_samples(features_scaled)[0]
        
        # Identify which features are anomalous
        anomalous_features = self._identify_anomalous_features(features)
        
        details = {
            "anomaly_score": float(score),
            "threshold": self.metadata.get("score_threshold", 0),
            "is_anomaly": bool(is_anomaly),
            "anomalous_features": anomalous_features,
            "confidence": self._score_to_confidence(score)
        }
        
        return is_anomaly, float(score), details
    
    def _identify_anomalous_features(
        self,
        features: np.ndarray
    ) -> List[Dict]:
        """Identify which features are unusual"""
        anomalous = []
        feature_stats = self.metadata.get("feature_stats", {})
        
        for i, name in enumerate(self.feature_names):
            if name not in feature_stats:
                continue
            
            stats = feature_stats[name]
            value = features[i] if len(features.shape) == 1 else features[0, i]
            
            # Calculate z-score
            if stats["std"] > 0:
                z_score = (value - stats["mean"]) / stats["std"]
            else:
                z_score = 0
            
            # Flag if > 2 standard deviations
            if abs(z_score) > 2:
                anomalous.append({
                    "feature": name,
                    "value": float(value),
                    "expected_mean": stats["mean"],
                    "z_score": float(z_score),
                    "direction": "high" if z_score > 0 else "low"
                })
        
        # Sort by absolute z-score
        anomalous.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        
        return anomalous[:5]  # Top 5 anomalous features
    
    def _score_to_confidence(self, score: float) -> str:
        """Convert anomaly score to confidence level"""
        threshold = self.metadata.get("score_threshold", 0)
        
        if score > threshold * 1.5:
            return "normal"
        elif score > threshold:
            return "slightly_unusual"
        elif score > threshold * 0.5:
            return "unusual"
        else:
            return "highly_anomalous"
    
    def predict_from_trip_features(
        self,
        trip_features: 'TripFeatures'
    ) -> Tuple[bool, float, Dict]:
        """Detect anomaly from TripFeatures object"""
        collector = DataCollector()
        example = collector.features_to_training_example(trip_features)
        X, _ = collector.prepare_features_matrix([example])
        return self.predict(X[0])
    
    def save(self, filename: str="anomaly_detector"):
        """Save trained model to disk"""
        if not self.is_trained:
            print("❌ No trained model to save")
            return
        
        model_path = self.model_dir / f"{filename}.joblib"
        scaler_path = self.model_dir / f"{filename}_scaler.joblib"
        metadata_path = self.model_dir / f"{filename}_metadata.json"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Model saved to {self.model_dir}")
    
    def load(self, filename: str="anomaly_detector") -> bool:
        """Load trained model from disk"""
        model_path = self.model_dir / f"{filename}.joblib"
        scaler_path = self.model_dir / f"{filename}_scaler.joblib"
        metadata_path = self.model_dir / f"{filename}_metadata.json"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.is_trained = True
            self.feature_names = self.metadata.get("feature_names", [])
            
            print(f"✅ Model loaded (trained {self.metadata.get('trained_at', 'unknown')})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


# Create singleton
anomaly_detector = AnomalyDetector()
