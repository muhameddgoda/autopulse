"""
AutoPulse Driver Scoring Model
Calculates driver safety score and behavior classification

This is a hybrid model:
- Phase 1: Rule-based weighted scoring (current)
- Phase 2: ML model refinement (future enhancement)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math

from app.ml.config import (
    THRESHOLDS, SCORING_WEIGHTS, PENALTIES,
    BEHAVIOR_LABELS, RISK_LEVELS,
    INSIGHT_TEMPLATES, RECOMMENDATION_TEMPLATES
)
from app.ml.features.extractor import TripFeatures


class BehaviorLabel(Enum):
    EXEMPLARY = "exemplary"
    CALM = "calm"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    DANGEROUS = "dangerous"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComponentScore:
    """Score for a single scoring component"""
    name: str
    raw_score: float      # 0-100 before weighting
    weight: float         # Weight factor
    weighted_score: float # raw_score * weight
    penalties_applied: List[str] = None
    
    def __post_init__(self):
        if self.penalties_applied is None:
            self.penalties_applied = []


@dataclass
class DriverScore:
    """Complete driver scoring result"""
    # Overall score
    total_score: float              # 0-100
    behavior_label: BehaviorLabel
    risk_level: RiskLevel
    
    # Component breakdown
    components: Dict[str, ComponentScore]
    
    # Insights and recommendations
    insights: List[str]
    recommendations: List[str]
    
    # Metadata
    trip_id: str
    vehicle_id: str
    duration_seconds: int
    distance_km: float
    
    # Raw feature summary
    feature_summary: Dict


class DriverScorer:
    """
    Calculates driver safety scores based on trip features
    
    Scoring Philosophy:
    - Start at 100 (perfect score)
    - Deduct points for unsafe behaviors
    - Weight different factors by importance
    - Generate actionable insights
    """
    
    def __init__(self):
        self.weights = SCORING_WEIGHTS
        self.penalties = PENALTIES
        self.thresholds = THRESHOLDS
    
    def score_trip(self, features: TripFeatures) -> DriverScore:
        """
        Calculate driver score for a trip
        
        Args:
            features: Extracted trip features
            
        Returns:
            DriverScore with breakdown and insights
        """
        components = {}
        
        # Score each component
        components["harsh_braking"] = self._score_harsh_braking(features)
        components["harsh_acceleration"] = self._score_harsh_acceleration(features)
        components["speeding"] = self._score_speeding(features)
        components["speed_consistency"] = self._score_speed_consistency(features)
        components["rpm_efficiency"] = self._score_rpm_efficiency(features)
        components["throttle_smoothness"] = self._score_throttle_smoothness(features)
        components["idle_time"] = self._score_idle_time(features)
        components["engine_stress"] = self._score_engine_stress(features)
        
        # Calculate total weighted score
        total_score = sum(c.weighted_score for c in components.values())
        total_score = max(0, min(100, total_score))  # Clamp to 0-100
        
        # Determine behavior label and risk level
        behavior_label = self._get_behavior_label(total_score)
        risk_level = self._get_risk_level(total_score)
        
        # Generate insights and recommendations
        insights = self._generate_insights(features, components)
        recommendations = self._generate_recommendations(features, components, total_score)
        
        return DriverScore(
            total_score=round(total_score, 1),
            behavior_label=behavior_label,
            risk_level=risk_level,
            components=components,
            insights=insights,
            recommendations=recommendations,
            trip_id=features.trip_id,
            vehicle_id=features.vehicle_id,
            duration_seconds=features.duration_seconds,
            distance_km=features.distance_km,
            feature_summary=self._create_feature_summary(features)
        )
    
    def _score_harsh_braking(self, features: TripFeatures) -> ComponentScore:
        """Score based on harsh braking events"""
        weight = self.weights["harsh_braking"]
        
        # Start at 100, deduct for events
        raw_score = 100
        penalties = []
        
        # Deduct for harsh brakes
        if features.harsh_brake_count > 0:
            deduction = features.harsh_brake_count * self.penalties["harsh_brake_event"]
            raw_score -= deduction
            penalties.append(f"-{deduction} for {features.harsh_brake_count} harsh brakes")
        
        # Extra penalty for severe brakes
        if features.severe_brake_count > 0:
            deduction = features.severe_brake_count * self.penalties["severe_brake_event"]
            raw_score -= deduction
            penalties.append(f"-{deduction} for {features.severe_brake_count} severe brakes")
        
        # Consider rate (events per hour)
        if features.harsh_brake_rate > 10:  # More than 10 per hour is bad
            rate_penalty = min(20, (features.harsh_brake_rate - 10) * 2)
            raw_score -= rate_penalty
            penalties.append(f"-{rate_penalty:.0f} for high brake rate")
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="Harsh Braking",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _score_harsh_acceleration(self, features: TripFeatures) -> ComponentScore:
        """Score based on harsh acceleration events"""
        weight = self.weights["harsh_acceleration"]
        
        raw_score = 100
        penalties = []
        
        if features.harsh_accel_count > 0:
            deduction = features.harsh_accel_count * self.penalties["harsh_accel_event"]
            raw_score -= deduction
            penalties.append(f"-{deduction} for {features.harsh_accel_count} harsh accels")
        
        if features.severe_accel_count > 0:
            deduction = features.severe_accel_count * self.penalties["severe_accel_event"]
            raw_score -= deduction
            penalties.append(f"-{deduction} for {features.severe_accel_count} severe accels")
        
        # Penalize high max acceleration
        if features.max_acceleration_g > 0.5:
            g_penalty = min(15, (features.max_acceleration_g - 0.5) * 30)
            raw_score -= g_penalty
            penalties.append(f"-{g_penalty:.0f} for high G-force ({features.max_acceleration_g:.2f}g)")
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="Harsh Acceleration",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _score_speeding(self, features: TripFeatures) -> ComponentScore:
        """Score based on speeding behavior"""
        weight = self.weights["speeding"]
        
        raw_score = 100
        penalties = []
        
        # Penalize speeding percentage
        if features.speeding_percentage > self.penalties["speeding_percentage_threshold"]:
            excess = features.speeding_percentage - self.penalties["speeding_percentage_threshold"]
            deduction = min(50, excess * 2)  # -2 points per % over threshold
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for {features.speeding_percentage:.1f}% speeding")
        
        # Extra penalty for dangerous speeds
        if features.dangerous_speed_seconds > 0:
            minutes = features.dangerous_speed_seconds / 60
            deduction = min(30, minutes * self.penalties["dangerous_speed_per_minute"])
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for dangerous speed")
        
        # Penalize very high max speed
        if features.max_speed_kmh > 180:
            speed_penalty = min(20, (features.max_speed_kmh - 180) * 0.5)
            raw_score -= speed_penalty
            penalties.append(f"-{speed_penalty:.0f} for max speed {features.max_speed_kmh:.0f} km/h")
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="Speeding",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _score_speed_consistency(self, features: TripFeatures) -> ComponentScore:
        """Score based on smooth, consistent speed maintenance"""
        weight = self.weights["speed_consistency"]
        
        # Lower std dev = more consistent = better score
        # Typical std dev: 10-30 for normal driving
        raw_score = 100
        penalties = []
        
        if features.speed_std_dev > 20:
            excess = features.speed_std_dev - 20
            deduction = min(40, excess * 1.5)
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for inconsistent speed")
        
        # Bonus for very smooth driving (negative penalty = bonus)
        if features.speed_std_dev < 10 and features.moving_seconds > 60:
            raw_score = min(100, raw_score + 10)
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="Speed Consistency",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _score_rpm_efficiency(self, features: TripFeatures) -> ComponentScore:
        """Score based on efficient RPM usage"""
        weight = self.weights["rpm_efficiency"]
        
        raw_score = 100
        penalties = []
        
        # Reward efficient RPM percentage
        if features.efficient_rpm_percentage < 50:
            deduction = (50 - features.efficient_rpm_percentage) * 0.5
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for inefficient RPM")
        
        # Penalize over-revving
        if features.over_rpm_percentage > 5:
            deduction = min(30, features.over_rpm_percentage * 2)
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for over-revving")
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="RPM Efficiency",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _score_throttle_smoothness(self, features: TripFeatures) -> ComponentScore:
        """Score based on smooth throttle usage"""
        weight = self.weights["throttle_smoothness"]
        
        raw_score = 100
        penalties = []
        
        # High throttle std dev = jerky driving
        if features.throttle_std_dev > 25:
            deduction = min(30, (features.throttle_std_dev - 25) * 1.5)
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for jerky throttle")
        
        # Penalize excessive high throttle
        if features.high_throttle_percentage > 20:
            deduction = min(25, (features.high_throttle_percentage - 20) * 1.25)
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for aggressive throttle")
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="Throttle Smoothness",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _score_idle_time(self, features: TripFeatures) -> ComponentScore:
        """Score based on idle time (fuel waste)"""
        weight = self.weights["idle_time"]
        
        raw_score = 100
        penalties = []
        
        if features.idle_percentage > self.penalties["idle_percentage_threshold"]:
            excess = features.idle_percentage - self.penalties["idle_percentage_threshold"]
            deduction = min(40, excess * 2)
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for {features.idle_percentage:.1f}% idle time")
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="Idle Time",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _score_engine_stress(self, features: TripFeatures) -> ComponentScore:
        """Score based on engine stress indicators"""
        weight = self.weights["engine_stress"]
        
        raw_score = 100
        penalties = []
        
        # High average engine stress
        if features.engine_stress_avg > 50:
            deduction = min(30, (features.engine_stress_avg - 50) * 0.6)
            raw_score -= deduction
            penalties.append(f"-{deduction:.0f} for engine stress")
        
        # High max engine temp
        if features.max_engine_temp > self.thresholds["high_engine_temp"]:
            temp_penalty = min(20, (features.max_engine_temp - self.thresholds["high_engine_temp"]) * 2)
            raw_score -= temp_penalty
            penalties.append(f"-{temp_penalty:.0f} for high temp")
        
        raw_score = max(0, raw_score)
        
        return ComponentScore(
            name="Engine Stress",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            penalties_applied=penalties
        )
    
    def _get_behavior_label(self, score: float) -> BehaviorLabel:
        """Determine behavior label from score"""
        for label, (min_score, max_score) in BEHAVIOR_LABELS.items():
            if min_score <= score <= max_score:
                return BehaviorLabel(label)
        return BehaviorLabel.NORMAL
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score"""
        for level, (min_score, max_score) in RISK_LEVELS.items():
            if min_score <= score <= max_score:
                return RiskLevel(level)
        return RiskLevel.MEDIUM
    
    def _generate_insights(
        self,
        features: TripFeatures,
        components: Dict[str, ComponentScore]
    ) -> List[str]:
        """Generate human-readable insights"""
        insights = []
        
        # Harsh braking insights
        if features.harsh_brake_count == 0:
            insights.append("No harsh braking events - excellent control")
        elif features.harsh_brake_count <= 2:
            insights.append(f"{features.harsh_brake_count} harsh braking events - good control")
        elif features.harsh_brake_count <= 5:
            insights.append(f"{features.harsh_brake_count} harsh braking events detected")
        else:
            insights.append(f"{features.harsh_brake_count} harsh braking events - needs improvement")
        
        # Harsh acceleration insights
        if features.harsh_accel_count == 0:
            insights.append("Smooth acceleration patterns")
        elif features.harsh_accel_count <= 3:
            insights.append(f"{features.harsh_accel_count} rapid acceleration events")
        else:
            insights.append(f"{features.harsh_accel_count} aggressive acceleration events")
        
        # Speeding insights
        if features.speeding_percentage == 0:
            insights.append("No speeding detected - excellent speed compliance")
        elif features.speeding_percentage < 10:
            insights.append(f"Minor speeding ({features.speeding_percentage:.1f}% of trip)")
        elif features.speeding_percentage < 25:
            insights.append(f"Speeding {features.speeding_percentage:.1f}% of trip time")
        else:
            insights.append(f"Excessive speeding ({features.speeding_percentage:.1f}%) - safety risk")
        
        # Max speed insight
        if features.max_speed_kmh > 0:
            insights.append(f"Top speed: {features.max_speed_kmh:.0f} km/h")
        
        # Idle insights
        if features.idle_percentage > 20:
            minutes = features.idle_seconds / 60
            insights.append(f"High idle time: {minutes:.1f} minutes ({features.idle_percentage:.1f}%)")
        
        # G-force insights
        if features.max_deceleration_g > 0.5:
            insights.append(f"Max braking force: {features.max_deceleration_g:.2f}g")
        if features.max_acceleration_g > 0.4:
            insights.append(f"Max acceleration: {features.max_acceleration_g:.2f}g")
        
        return insights
    
    def _generate_recommendations(
        self,
        features: TripFeatures,
        components: Dict[str, ComponentScore],
        total_score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Only add recommendations for areas needing improvement
        if components["harsh_braking"].raw_score < 70:
            recommendations.append(RECOMMENDATION_TEMPLATES["harsh_braking"])
        
        if components["harsh_acceleration"].raw_score < 70:
            recommendations.append(RECOMMENDATION_TEMPLATES["harsh_acceleration"])
        
        if components["speeding"].raw_score < 70:
            if features.dangerous_speed_seconds > 0:
                recommendations.append(RECOMMENDATION_TEMPLATES["dangerous_speed"])
            else:
                recommendations.append(RECOMMENDATION_TEMPLATES["speeding"])
        
        if components["rpm_efficiency"].raw_score < 70:
            recommendations.append(RECOMMENDATION_TEMPLATES["over_rpm"])
        
        if components["idle_time"].raw_score < 70:
            recommendations.append(RECOMMENDATION_TEMPLATES["high_idle"])
        
        if components["throttle_smoothness"].raw_score < 70:
            recommendations.append(RECOMMENDATION_TEMPLATES["aggressive_throttle"])
        
        if components["engine_stress"].raw_score < 70:
            recommendations.append(RECOMMENDATION_TEMPLATES["engine_stress"])
        
        # Add positive reinforcement if score is good
        if total_score >= 80 and not recommendations:
            recommendations.append("Keep up the great driving! Your habits are safe and efficient.")
        
        return recommendations
    
    def _create_feature_summary(self, features: TripFeatures) -> Dict:
        """Create a summary of key features"""
        return {
            "duration_minutes": round(features.duration_seconds / 60, 1),
            "distance_km": round(features.distance_km, 2),
            "avg_speed_kmh": round(features.avg_speed_kmh, 1),
            "max_speed_kmh": round(features.max_speed_kmh, 1),
            "harsh_events_total": features.harsh_brake_count + features.harsh_accel_count,
            "speeding_percentage": round(features.speeding_percentage, 1),
            "idle_percentage": round(features.idle_percentage, 1),
        }
    
    def score_to_dict(self, score: DriverScore) -> Dict:
        """Convert DriverScore to dictionary for JSON serialization"""
        return {
            "score": score.total_score,
            "behavior": score.behavior_label.value,
            "risk_level": score.risk_level.value,
            
            "components": {
                name: {
                    "score": round(comp.raw_score, 1),
                    "weight": comp.weight,
                    "weighted": round(comp.weighted_score, 2),
                }
                for name, comp in score.components.items()
            },
            
            "insights": score.insights,
            "recommendations": score.recommendations,
            
            "trip_info": {
                "trip_id": score.trip_id,
                "vehicle_id": score.vehicle_id,
                "duration_seconds": score.duration_seconds,
                "distance_km": round(score.distance_km, 2),
            },
            
            "summary": score.feature_summary,
        }


# Create singleton instance
scorer = DriverScorer()
