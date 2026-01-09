"""
AutoPulse Hybrid Scorer
Combines rule-based scoring with ML predictions

Strategy:
1. Always calculate rule-based score (reliable baseline)
2. If ML models are trained, enhance with predictions
3. Use ML for behavior classification and anomaly detection
4. Keep rules for interpretable component scores
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from app.ml.features.extractor import TripFeatures, FeatureExtractor
from app.ml.models.driver_scorer import DriverScorer, DriverScore
from app.ml.training.data_collector import DataCollector
from app.ml.training.behavior_classifier import BehaviorClassifier
from app.ml.training.anomaly_detector import AnomalyDetector


@dataclass
class HybridScore:
    """Combined rule-based and ML score"""
    # Rule-based results
    rule_score: float
    rule_behavior: str
    rule_risk: str
    
    # ML results (if available)
    ml_behavior: Optional[str] = None
    ml_behavior_confidence: Optional[Dict[str, float]] = None
    ml_is_anomaly: Optional[bool] = None
    ml_anomaly_score: Optional[float] = None
    ml_anomaly_details: Optional[Dict] = None
    
    # Combined results
    final_score: float = 0
    final_behavior: str = ""
    final_risk: str = ""
    
    # Metadata
    ml_available: bool = False
    using_ml: bool = False


class HybridScorer:
    """
    Combines rule-based scoring with ML predictions
    """
    
    def __init__(self, model_dir: str=None):
        self.model_dir = model_dir
        self.rule_scorer = DriverScorer()
        self.feature_extractor = FeatureExtractor()
        self.data_collector = DataCollector()
        
        # ML models (use their default paths if model_dir not specified)
        self.behavior_classifier = BehaviorClassifier(model_dir) if model_dir else BehaviorClassifier()
        self.anomaly_detector = AnomalyDetector(model_dir) if model_dir else AnomalyDetector()
        
        # Will be set when loading for specific vehicle
        self.ml_behavior_available = False
        self.ml_anomaly_available = False
        self._loaded_vehicle_id = None
    
    def load_models_for_vehicle(self, vehicle_id: str, force_reload: bool=False) -> bool:
        """Load trained models for a specific vehicle"""
        if self._loaded_vehicle_id == vehicle_id and not force_reload:
            return self.ml_behavior_available or self.ml_anomaly_available
        
        # Reset state for fresh load
        self.ml_behavior_available = False
        self.ml_anomaly_available = False
        
        # Try vehicle-specific models first
        self.ml_behavior_available = self.behavior_classifier.load(f"behavior_{vehicle_id}")
        self.ml_anomaly_available = self.anomaly_detector.load(f"anomaly_{vehicle_id}")
        
        # Fallback to generic models
        if not self.ml_behavior_available:
            self.ml_behavior_available = self.behavior_classifier.load("behavior_classifier")
        if not self.ml_anomaly_available:
            self.ml_anomaly_available = self.anomaly_detector.load("anomaly_detector")
        
        self._loaded_vehicle_id = vehicle_id
        return self.ml_behavior_available or self.ml_anomaly_available
    
    def score_trip(
        self,
        features: TripFeatures,
        use_ml: bool=True
    ) -> HybridScore:
        """
        Score a trip using rules and ML
        
        Args:
            features: Extracted trip features
            use_ml: Whether to use ML models if available
            
        Returns:
            HybridScore with combined results
        """
        # 1. Always get rule-based score
        rule_result = self.rule_scorer.score_trip(features)
        
        result = HybridScore(
            rule_score=rule_result.total_score,
            rule_behavior=rule_result.behavior_label.value,
            rule_risk=rule_result.risk_level.value,
            final_score=rule_result.total_score,
            final_behavior=rule_result.behavior_label.value,
            final_risk=rule_result.risk_level.value,
            ml_available=self.ml_behavior_available or self.ml_anomaly_available
        )
        
        if not use_ml:
            return result
        
        # 2. ML Behavior Classification
        if self.ml_behavior_available:
            try:
                ml_behavior, ml_proba = self.behavior_classifier.predict_from_trip_features(features)
                result.ml_behavior = ml_behavior
                result.ml_behavior_confidence = ml_proba
                result.using_ml = True
            except Exception as e:
                print(f"ML behavior prediction error: {e}")
        
        # 3. ML Anomaly Detection
        if self.ml_anomaly_available:
            try:
                is_anomaly, anomaly_score, details = self.anomaly_detector.predict_from_trip_features(features)

                result.ml_is_anomaly = bool(is_anomaly)
                result.ml_anomaly_score = float(anomaly_score) if anomaly_score is not None else None

                # sanitize dict recursively if needed
                result.ml_anomaly_details = self._sanitize(details)

                result.using_ml = True
            except Exception as e:
                print(f"ML anomaly detection error: {e}")
        
        # 4. Combine results
        result = self._combine_scores(result)
        
        return result
    
    def _combine_scores(self, result: HybridScore) -> HybridScore:
        """
        Combine rule-based and ML results into final score
        
        Strategy:
        - Start with rule-based score
        - Use ML behavior if confidence is high (>60%)
        - Adjust score based on ML vs Rule disagreement
        - Flag anomalies for review
        """
        # Start with rule-based
        result.final_score = result.rule_score
        result.final_behavior = result.rule_behavior
        result.final_risk = result.rule_risk
        
        # Use ML behavior based on highest confidence class
        if result.ml_behavior_confidence:
            # Find the class with highest confidence
            best_class = max(result.ml_behavior_confidence.items(), key=lambda x: x[1])
            ml_behavior = best_class[0]
            ml_confidence = best_class[1]
            
            # Update ml_behavior to match highest confidence
            result.ml_behavior = ml_behavior
            
            if ml_confidence > 0.6:
                result.final_behavior = ml_behavior
                
                # Adjust score based on ML behavior
                # ML "calm" but rule says "normal" → boost score slightly
                # ML "aggressive" but rule says "normal" → reduce score
                behavior_order = ['dangerous', 'aggressive', 'normal', 'calm', 'exemplary']
                
                try:
                    ml_idx = behavior_order.index(ml_behavior)
                    rule_idx = behavior_order.index(result.rule_behavior)
                    
                    # Positive diff = ML thinks better, negative = ML thinks worse
                    diff = ml_idx - rule_idx
                    
                    # Blend: adjust score by up to 10 points based on ML confidence
                    adjustment = diff * 5 * ml_confidence
                    result.final_score = max(0, min(100, result.final_score + adjustment))
                except ValueError:
                    pass  # Unknown behavior, skip adjustment
                
                # Adjust risk based on ML behavior
                if ml_behavior in ["aggressive", "dangerous"]:
                    if result.rule_risk == "low":
                        result.final_risk = "medium"
                    elif result.rule_risk == "medium":
                        result.final_risk = "high"
                elif ml_behavior in ["calm", "exemplary"]:
                    if result.rule_risk == "high":
                        result.final_risk = "medium"
                    elif result.rule_risk == "medium" and ml_confidence > 0.7:
                        result.final_risk = "low"
        
        # If anomaly detected, increase risk level
        if result.ml_is_anomaly:
            if result.final_risk == "low":
                result.final_risk = "medium"
            elif result.final_risk == "medium":
                result.final_risk = "high"
            
            # Reduce score slightly for anomalies
            result.final_score = max(0, result.final_score - 5)
        
        return result
    
    def score_to_dict(self, score: HybridScore, rule_result: DriverScore=None) -> Dict:
        """Convert HybridScore to dictionary"""
        result = {
            "score": score.final_score,
            "behavior": score.final_behavior,
            "risk_level": score.final_risk,
            
            "rule_based": {
                "score": score.rule_score,
                "behavior": score.rule_behavior,
                "risk": score.rule_risk
            },
            
            "ml_enhanced": score.using_ml,
            "ml_available": score.ml_available,
        }
        
        if score.ml_behavior:
            result["ml_behavior"] = {
                "prediction": score.ml_behavior,
                "confidence": score.ml_behavior_confidence
            }
        
        if score.ml_is_anomaly is not None:
            result["ml_anomaly"] = {
                "is_anomaly": bool(score.ml_is_anomaly),
                "score": float(score.ml_anomaly_score) if score.ml_anomaly_score is not None else None,
                "details": score.ml_anomaly_details
            }
        
        return result
    
    def train_models(
        self,
        examples: list,
        train_behavior: bool=True,
        train_anomaly: bool=True
    ) -> Dict:
        """
        Train both ML models
        
        Args:
            examples: List of TrainingExample objects
            train_behavior: Whether to train behavior classifier
            train_anomaly: Whether to train anomaly detector
            
        Returns:
            Training results for each model
        """
        results = {}
        
        if train_behavior:
            print("\n📊 Training Behavior Classifier...")
            behavior_result = self.behavior_classifier.train(examples)
            if behavior_result.get("success"):
                self.behavior_classifier.save()
                self.ml_behavior_available = True
            results["behavior_classifier"] = behavior_result
        
        if train_anomaly:
            print("\n🔍 Training Anomaly Detector...")
            anomaly_result = self.anomaly_detector.train(examples)
            if anomaly_result.get("success"):
                self.anomaly_detector.save()
                self.ml_anomaly_available = True
            results["anomaly_detector"] = anomaly_result
        
        return results

    def _sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj


# Create singleton
hybrid_scorer = HybridScorer()
