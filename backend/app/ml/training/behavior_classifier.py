"""
AutoPulse Behavior Classifier
Trains ML model to classify driving behavior: calm, normal, aggressive

Algorithms:
- Random Forest (default, good interpretability)
- XGBoost (optional, better performance)
- Gradient Boosting (fallback)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Run: pip install scikit-learn")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from app.ml.training.data_collector import DataCollector, TrainingExample


class BehaviorClassifier:
    """
    ML classifier for driving behavior
    
    Classes:
    - exemplary: Excellent, very safe driving
    - calm: Safe, defensive driving
    - normal: Average driving patterns
    - aggressive: Risky, aggressive behavior
    - dangerous: Very dangerous driving
    """
    
    CLASSES = ['exemplary', 'calm', 'normal', 'aggressive', 'dangerous']
    
    # Compute default model directory relative to this file
    _DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "trained_models"
    
    def __init__(self, model_dir: str=None):
        self.model_dir = Path(model_dir) if model_dir else self._DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.trained_classes: List[str] = []  # Classes actually used in training
        
        # Training metadata
        self.metadata = {
            "trained_at": None,
            "n_samples": 0,
            "accuracy": 0,
            "algorithm": "",
            "feature_importance": {}
        }
    
    def train(
        self,
        examples: List[TrainingExample],
        algorithm: str="random_forest",
        test_size: float=0.2,
        random_state: int=42
    ) -> Dict:
        """
        Train the behavior classifier
        
        Args:
            examples: List of labeled training examples
            algorithm: 'random_forest', 'xgboost', or 'gradient_boosting'
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Training results with metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed"}
        
        if len(examples) < 10:
            return {"error": f"Need at least 10 examples, got {len(examples)}"}
        
        # Prepare data
        collector = DataCollector()
        X, self.feature_names = collector.prepare_features_matrix(examples)
        
        # Get labels
        y_labels = [ex.behavior_label for ex in examples]
        
        # Check if we have labeled data
        labeled_count = sum(1 for label in y_labels if label in self.CLASSES)
        if labeled_count < 10:
            # Auto-label using rule-based scores
            print("⚠️ Not enough labeled data. Auto-labeling from rule-based scores...")
            y_labels = self._auto_label_from_scores(examples)
        
        # Encode labels - fit on actual labels present in data
        unique_labels = list(set(y_labels))
        self.label_encoder.fit(unique_labels)
        y = self.label_encoder.transform(y_labels)
        
        # Store actual classes used
        self.trained_classes = unique_labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (disable stratify if too few samples per class)
        unique, counts = np.unique(y, return_counts=True)
        min_samples_per_class = min(counts)
        
        if min_samples_per_class >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # Not enough samples to stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
        
        # Select algorithm
        if algorithm == "xgboost" and XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif algorithm == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            # Default: Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            )
        
        # Train
        print(f"🚀 Training {algorithm} classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
            # Sort by importance
            importance = dict(sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        else:
            importance = {}
        
        # Update metadata
        self.metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(examples),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": float(accuracy),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "algorithm": algorithm,
            "feature_importance": importance,
            "classes": self.trained_classes,  # Use actual trained classes
        }
        
        self.is_trained = True
        
        # Detailed report - use only classes present in test set
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        present_classes = [self.trained_classes[i] for i in unique_labels if i < len(self.trained_classes)]
        
        report = classification_report(
            y_test, y_pred,
            target_names=present_classes,
            labels=unique_labels,
            output_dict=True
        )
        
        results = {
            "success": True,
            "accuracy": accuracy,
            "cv_scores": {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores.tolist()
            },
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importance": importance,
            "top_features": list(importance.keys())[:5] if importance else [],
            "metadata": self.metadata
        }
        
        print(f"✅ Training complete! Accuracy: {accuracy:.2%}")
        print(f"   Cross-validation: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
        
        return results
    
    def _auto_label_from_scores(
        self,
        examples: List[TrainingExample]
    ) -> List[str]:
        """
        Auto-generate labels from rule-based scores
        
        Score ranges:
        - 75-100: calm
        - 50-74: normal
        - 0-49: aggressive
        """
        labels = []
        for ex in examples:
            score = ex.rule_based_score
            if score >= 75:
                labels.append('calm')
            elif score >= 50:
                labels.append('normal')
            else:
                labels.append('aggressive')
        
        # Log distribution
        from collections import Counter
        dist = Counter(labels)
        print(f"   Label distribution: {dict(dist)}")
        
        return labels
    
    def predict(self, features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Predict behavior for a single trip
        
        Args:
            features: Feature array (1, n_features)
            
        Returns:
            (predicted_label, probability_dict)
        """
        if not self.is_trained or self.model is None:
            return "normal", {"calm": 0.33, "normal": 0.34, "aggressive": 0.33}
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        pred_idx = self.model.predict(features_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            classes_to_use = self.trained_classes if self.trained_classes else self.CLASSES
            proba_dict = {
                classes_to_use[i]: float(proba[i])
                for i in range(min(len(classes_to_use), len(proba)))
            }
        else:
            proba_dict = {pred_label: 1.0}
        
        return pred_label, proba_dict
    
    def predict_from_trip_features(
        self,
        trip_features: 'TripFeatures'
    ) -> Tuple[str, Dict[str, float]]:
        """Predict from TripFeatures object"""
        collector = DataCollector()
        example = collector.features_to_training_example(trip_features)
        X, _ = collector.prepare_features_matrix([example])
        return self.predict(X[0])
    
    def save(self, filename: str="behavior_classifier"):
        """Save trained model to disk"""
        if not self.is_trained:
            print("❌ No trained model to save")
            return
        
        model_path = self.model_dir / f"{filename}.joblib"
        scaler_path = self.model_dir / f"{filename}_scaler.joblib"
        encoder_path = self.model_dir / f"{filename}_encoder.joblib"
        metadata_path = self.model_dir / f"{filename}_metadata.json"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Model saved to {self.model_dir}")
    
    def load(self, filename: str="behavior_classifier") -> bool:
        """Load trained model from disk"""
        model_path = self.model_dir / f"{filename}.joblib"
        scaler_path = self.model_dir / f"{filename}_scaler.joblib"
        encoder_path = self.model_dir / f"{filename}_encoder.joblib"
        metadata_path = self.model_dir / f"{filename}_metadata.json"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.is_trained = True
            self.feature_names = list(self.metadata.get('feature_importance', {}).keys())
            self.trained_classes = self.metadata.get('classes', self.CLASSES)
            
            print(f"✅ Model loaded (trained {self.metadata.get('trained_at', 'unknown')})")
            print(f"   Accuracy: {self.metadata.get('accuracy', 0):.2%}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


# Create singleton
behavior_classifier = BehaviorClassifier()
