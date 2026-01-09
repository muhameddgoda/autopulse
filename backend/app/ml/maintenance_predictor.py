"""
AutoPulse Predictive Maintenance
Predicts component wear and maintenance needs based on driving behavior

Components tracked:
- Brake pads (wear from hard braking)
- Engine oil (degradation from stress/mileage)
- Tires (wear from aggressive driving)
- Battery (health from usage patterns)
- Air filter (based on mileage)
- Transmission fluid (based on harsh shifts)

Uses XGBoost for failure prediction when enough data is available,
falls back to rule-based estimation otherwise.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ComponentStatus(Enum):
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MaintenanceUrgency(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMEDIATE = "immediate"


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    component: str
    health_percentage: float  # 0-100
    status: ComponentStatus
    urgency: MaintenanceUrgency
    prediction: str  # Human-readable prediction
    estimated_remaining_km: float
    estimated_remaining_days: int
    next_service: str
    factors: List[str]  # Contributing factors
    confidence: float  # 0-1 confidence in prediction


@dataclass
class MaintenancePrediction:
    """Complete maintenance prediction for a vehicle"""
    vehicle_id: str
    prediction_date: datetime
    overall_health: float  # 0-100
    components: Dict[str, ComponentHealth]
    urgent_items: List[str]
    recommendations: List[str]
    estimated_next_service_km: float
    estimated_next_service_date: str


class MaintenancePredictor:
    """
    Predicts maintenance needs based on driving behavior and vehicle data
    
    Uses a combination of:
    1. Rule-based heuristics (always available)
    2. ML regression models (when trained)
    
    Key factors:
    - Harsh braking events -> brake wear
    - Aggressive acceleration -> engine/transmission stress
    - High RPM usage -> engine wear
    - Total mileage -> general wear
    - Sport mode percentage -> overall component stress
    """
    
    # Default service intervals (km)
    SERVICE_INTERVALS = {
        "brake_pads": 50000,
        "engine_oil": 15000,
        "tires": 40000,
        "battery": 100000,  # or ~4-5 years
        "air_filter": 30000,
        "transmission_fluid": 60000,
        "spark_plugs": 80000,
        "coolant": 50000,
    }
    
    # Wear multipliers based on driving behavior
    WEAR_MULTIPLIERS = {
        "aggressive": 1.5,
        "sport": 1.3,
        "normal": 1.0,
        "calm": 0.85,
        "exemplary": 0.75,
    }
    
    def __init__(self, model_dir: str = "app/ml/trained_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # ML models for each component
        self.models: Dict[str, any] = {}
        self.scalers: Dict[str, any] = {}
        self.is_trained: Dict[str, bool] = {}
        
        # Vehicle state (would normally come from database)
        self.vehicle_mileage: Dict[str, float] = {}
        self.last_service: Dict[str, Dict[str, datetime]] = {}
    
    def predict_maintenance(
        self,
        vehicle_id: str,
        driving_summary: Dict,
        current_mileage: float = 0,
        last_services: Optional[Dict[str, datetime]] = None
    ) -> MaintenancePrediction:
        """
        Generate maintenance predictions for a vehicle
        
        Args:
            vehicle_id: Vehicle identifier
            driving_summary: Summary of driving behavior (from ML service)
            current_mileage: Current odometer reading (km)
            last_services: Dict of component -> last service date
            
        Returns:
            MaintenancePrediction with all component health estimates
        """
        if last_services is None:
            last_services = {}
        
        # Extract driving behavior metrics
        behavior = driving_summary.get("overall_behavior", "normal")
        avg_score = driving_summary.get("score_statistics", {}).get("average", 70)
        totals = driving_summary.get("totals", {})
        events_per_100km = driving_summary.get("events_per_100km", {})
        
        harsh_brakes_rate = events_per_100km.get("harsh_brakes", 0)
        harsh_accels_rate = events_per_100km.get("harsh_accels", 0)
        total_distance = totals.get("distance_km", 0)
        total_harsh_brakes = totals.get("harsh_brakes", 0)
        total_harsh_accels = totals.get("harsh_accelerations", 0)
        
        # Get wear multiplier based on behavior
        wear_mult = self.WEAR_MULTIPLIERS.get(behavior, 1.0)
        
        # Predict each component
        components = {}
        
        # Brake Pads
        components["brake_pads"] = self._predict_brake_pads(
            harsh_brakes_rate, total_harsh_brakes, total_distance,
            current_mileage, wear_mult, last_services.get("brake_pads")
        )
        
        # Engine Oil
        components["engine_oil"] = self._predict_engine_oil(
            avg_score, total_distance, current_mileage,
            wear_mult, last_services.get("engine_oil")
        )
        
        # Tires
        components["tires"] = self._predict_tires(
            behavior, harsh_brakes_rate, harsh_accels_rate,
            total_distance, current_mileage, wear_mult,
            last_services.get("tires")
        )
        
        # Battery
        components["battery"] = self._predict_battery(
            current_mileage, last_services.get("battery")
        )
        
        # Air Filter
        components["air_filter"] = self._predict_air_filter(
            current_mileage, last_services.get("air_filter")
        )
        
        # Transmission Fluid
        components["transmission_fluid"] = self._predict_transmission(
            harsh_accels_rate, total_distance, current_mileage,
            wear_mult, last_services.get("transmission_fluid")
        )
        
        # Calculate overall health
        health_values = [c.health_percentage for c in components.values()]
        overall_health = np.mean(health_values) if health_values else 100
        
        # Find urgent items
        urgent_items = [
            name for name, comp in components.items()
            if comp.urgency in [MaintenanceUrgency.HIGH, MaintenanceUrgency.IMMEDIATE]
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(components, behavior, avg_score)
        
        # Estimate next service
        min_remaining_km = min(
            (c.estimated_remaining_km for c in components.values()),
            default=10000
        )
        
        # Estimate date based on average daily driving
        avg_daily_km = total_distance / max(1, driving_summary.get("total_trips", 1) / 7 * 30)
        days_to_service = int(min_remaining_km / max(1, avg_daily_km))
        next_service_date = (datetime.now() + timedelta(days=days_to_service)).strftime("%Y-%m-%d")
        
        return MaintenancePrediction(
            vehicle_id=vehicle_id,
            prediction_date=datetime.now(),
            overall_health=round(overall_health, 1),
            components=components,
            urgent_items=urgent_items,
            recommendations=recommendations,
            estimated_next_service_km=round(min_remaining_km, 0),
            estimated_next_service_date=next_service_date
        )
    
    def _predict_brake_pads(
        self,
        harsh_rate: float,
        total_harsh: int,
        recent_distance: float,
        mileage: float,
        wear_mult: float,
        last_service: Optional[datetime]
    ) -> ComponentHealth:
        """Predict brake pad wear"""
        base_interval = self.SERVICE_INTERVALS["brake_pads"]
        
        # Adjust interval based on driving
        # More harsh braking = faster wear
        brake_factor = 1 + (harsh_rate / 20)  # Each 20 events/100km doubles wear
        adjusted_interval = base_interval / (brake_factor * wear_mult)
        
        # Estimate current wear
        km_since_service = mileage if last_service is None else min(mileage, recent_distance * 4)
        wear_percentage = (km_since_service / adjusted_interval) * 100
        health = max(0, 100 - wear_percentage)
        
        # Determine status
        if health > 60:
            status = ComponentStatus.GOOD
            urgency = MaintenanceUrgency.NONE
        elif health > 30:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.MEDIUM
        elif health > 10:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.HIGH
        else:
            status = ComponentStatus.CRITICAL
            urgency = MaintenanceUrgency.IMMEDIATE
        
        remaining_km = max(0, adjusted_interval - km_since_service)
        
        factors = []
        if harsh_rate > 10:
            factors.append(f"High hard braking rate ({harsh_rate:.1f}/100km)")
        if wear_mult > 1.2:
            factors.append("Aggressive driving style detected")
        if not factors:
            factors.append("Normal wear pattern")
        
        prediction = self._get_brake_prediction(health, harsh_rate)
        
        return ComponentHealth(
            component="Brake Pads",
            health_percentage=round(health, 1),
            status=status,
            urgency=urgency,
            prediction=prediction,
            estimated_remaining_km=round(remaining_km, 0),
            estimated_remaining_days=int(remaining_km / 50) if remaining_km > 0 else 0,
            next_service=f"~{int(remaining_km):,} km",
            factors=factors,
            confidence=0.75
        )
    
    def _predict_engine_oil(
        self,
        avg_score: float,
        recent_distance: float,
        mileage: float,
        wear_mult: float,
        last_service: Optional[datetime]
    ) -> ComponentHealth:
        """Predict engine oil degradation"""
        base_interval = self.SERVICE_INTERVALS["engine_oil"]
        
        # Aggressive driving degrades oil faster
        stress_factor = 1 + ((100 - avg_score) / 100)  # Lower score = more stress
        adjusted_interval = base_interval / (stress_factor * wear_mult)
        
        km_since_service = mileage if last_service is None else min(mileage, recent_distance * 4)
        degradation = (km_since_service / adjusted_interval) * 100
        health = max(0, 100 - degradation)
        
        if health > 50:
            status = ComponentStatus.GOOD
            urgency = MaintenanceUrgency.NONE
        elif health > 25:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.MEDIUM
        elif health > 10:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.HIGH
        else:
            status = ComponentStatus.CRITICAL
            urgency = MaintenanceUrgency.IMMEDIATE
        
        remaining_km = max(0, adjusted_interval - km_since_service)
        
        factors = []
        if avg_score < 60:
            factors.append("High engine stress from driving style")
        if wear_mult > 1.2:
            factors.append("Sport/aggressive driving accelerates oil breakdown")
        if not factors:
            factors.append("Normal degradation rate")
        
        prediction = "Oil change recommended soon" if health < 30 else "Oil condition nominal"
        
        return ComponentHealth(
            component="Engine Oil",
            health_percentage=round(health, 1),
            status=status,
            urgency=urgency,
            prediction=prediction,
            estimated_remaining_km=round(remaining_km, 0),
            estimated_remaining_days=int(remaining_km / 50) if remaining_km > 0 else 0,
            next_service=f"~{int(remaining_km):,} km",
            factors=factors,
            confidence=0.8
        )
    
    def _predict_tires(
        self,
        behavior: str,
        brake_rate: float,
        accel_rate: float,
        recent_distance: float,
        mileage: float,
        wear_mult: float,
        last_service: Optional[datetime]
    ) -> ComponentHealth:
        """Predict tire wear"""
        base_interval = self.SERVICE_INTERVALS["tires"]
        
        # Aggressive driving = faster tire wear
        combined_rate = brake_rate + accel_rate
        tire_factor = 1 + (combined_rate / 30)
        adjusted_interval = base_interval / (tire_factor * wear_mult)
        
        km_since_service = mileage if last_service is None else min(mileage, recent_distance * 4)
        wear = (km_since_service / adjusted_interval) * 100
        health = max(0, 100 - wear)
        
        if health > 50:
            status = ComponentStatus.GOOD
            urgency = MaintenanceUrgency.NONE
        elif health > 25:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.MEDIUM
        else:
            status = ComponentStatus.CRITICAL
            urgency = MaintenanceUrgency.HIGH
        
        remaining_km = max(0, adjusted_interval - km_since_service)
        
        factors = []
        if behavior in ["aggressive", "dangerous"]:
            factors.append("Aggressive driving increases tire wear")
        if combined_rate > 15:
            factors.append("Frequent hard maneuvers detected")
        if not factors:
            factors.append("Even wear pattern expected")
        
        prediction = "Check tire tread depth" if health < 40 else "Tires in good condition"
        
        return ComponentHealth(
            component="Tires",
            health_percentage=round(health, 1),
            status=status,
            urgency=urgency,
            prediction=prediction,
            estimated_remaining_km=round(remaining_km, 0),
            estimated_remaining_days=int(remaining_km / 50) if remaining_km > 0 else 0,
            next_service=f"~{int(remaining_km):,} km",
            factors=factors,
            confidence=0.7
        )
    
    def _predict_battery(
        self,
        mileage: float,
        last_service: Optional[datetime]
    ) -> ComponentHealth:
        """Predict battery health"""
        # Battery health is primarily time-based
        if last_service:
            days_since = (datetime.now() - last_service).days
            # Assume 4-year battery life
            health = max(0, 100 - (days_since / (4 * 365)) * 100)
        else:
            # Estimate from mileage (assume 15k km/year)
            estimated_years = mileage / 15000
            health = max(0, 100 - (estimated_years / 4) * 100)
        
        if health > 50:
            status = ComponentStatus.GOOD
            urgency = MaintenanceUrgency.NONE
        elif health > 25:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.LOW
        else:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.MEDIUM
        
        remaining_days = int((health / 100) * 4 * 365)
        
        return ComponentHealth(
            component="Battery",
            health_percentage=round(health, 1),
            status=status,
            urgency=urgency,
            prediction="Battery health nominal" if health > 40 else "Consider battery test",
            estimated_remaining_km=remaining_days * 40,  # Assume 40km/day
            estimated_remaining_days=remaining_days,
            next_service=f"~{remaining_days // 30} months",
            factors=["Age-based degradation"],
            confidence=0.65
        )
    
    def _predict_air_filter(
        self,
        mileage: float,
        last_service: Optional[datetime]
    ) -> ComponentHealth:
        """Predict air filter condition"""
        base_interval = self.SERVICE_INTERVALS["air_filter"]
        
        km_since = mileage  # Simplified - would use actual service records
        wear = (km_since / base_interval) * 100
        health = max(0, 100 - (wear % 100))  # Cycles every interval
        
        if health > 50:
            status = ComponentStatus.GOOD
            urgency = MaintenanceUrgency.NONE
        elif health > 25:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.LOW
        else:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.MEDIUM
        
        remaining = base_interval - (km_since % base_interval)
        
        return ComponentHealth(
            component="Air Filter",
            health_percentage=round(health, 1),
            status=status,
            urgency=urgency,
            prediction="Air filter replacement due soon" if health < 30 else "Air filter OK",
            estimated_remaining_km=round(remaining, 0),
            estimated_remaining_days=int(remaining / 50),
            next_service=f"~{int(remaining):,} km",
            factors=["Mileage-based replacement"],
            confidence=0.8
        )
    
    def _predict_transmission(
        self,
        accel_rate: float,
        recent_distance: float,
        mileage: float,
        wear_mult: float,
        last_service: Optional[datetime]
    ) -> ComponentHealth:
        """Predict transmission fluid condition"""
        base_interval = self.SERVICE_INTERVALS["transmission_fluid"]
        
        # Hard acceleration stresses transmission
        trans_factor = 1 + (accel_rate / 25)
        adjusted_interval = base_interval / (trans_factor * wear_mult)
        
        km_since = mileage if last_service is None else min(mileage, recent_distance * 4)
        wear = (km_since / adjusted_interval) * 100
        health = max(0, 100 - wear)
        
        if health > 50:
            status = ComponentStatus.GOOD
            urgency = MaintenanceUrgency.NONE
        elif health > 25:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.LOW
        else:
            status = ComponentStatus.WARNING
            urgency = MaintenanceUrgency.MEDIUM
        
        remaining = max(0, adjusted_interval - km_since)
        
        factors = []
        if accel_rate > 10:
            factors.append("Hard accelerations stress transmission")
        if not factors:
            factors.append("Normal transmission wear")
        
        return ComponentHealth(
            component="Transmission Fluid",
            health_percentage=round(health, 1),
            status=status,
            urgency=urgency,
            prediction="Fluid service recommended" if health < 30 else "Transmission fluid OK",
            estimated_remaining_km=round(remaining, 0),
            estimated_remaining_days=int(remaining / 50),
            next_service=f"~{int(remaining):,} km",
            factors=factors,
            confidence=0.7
        )
    
    def _get_brake_prediction(self, health: float, harsh_rate: float) -> str:
        """Generate brake pad prediction text"""
        if health > 70:
            if harsh_rate > 15:
                return "Good condition, but high braking frequency detected"
            return "Brake pads in good condition"
        elif health > 40:
            return "Moderate wear - monitor during next service"
        elif health > 20:
            return "Significant wear - schedule inspection soon"
        else:
            return "Brake pads need replacement"
    
    def _generate_recommendations(
        self,
        components: Dict[str, ComponentHealth],
        behavior: str,
        avg_score: float
    ) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []
        
        # Component-specific recommendations
        for name, comp in components.items():
            if comp.urgency == MaintenanceUrgency.IMMEDIATE:
                recommendations.append(f"⚠️ URGENT: {comp.component} needs immediate attention")
            elif comp.urgency == MaintenanceUrgency.HIGH:
                recommendations.append(f"Schedule {comp.component.lower()} service soon")
        
        # Behavior-based recommendations
        if behavior in ["aggressive", "dangerous"]:
            recommendations.append("Consider smoother driving to reduce component wear")
        
        if avg_score < 60:
            recommendations.append("Improving driving score could extend maintenance intervals by 20-30%")
        
        # Positive reinforcement
        if not recommendations and avg_score >= 75:
            recommendations.append("Great driving habits! Keep it up to maximize vehicle longevity")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def to_dict(self, prediction: MaintenancePrediction) -> Dict:
        """Convert prediction to JSON-serializable dict"""
        return {
            "vehicle_id": prediction.vehicle_id,
            "prediction_date": prediction.prediction_date.isoformat(),
            "overall_health": prediction.overall_health,
            "components": {
                name: {
                    "component": comp.component,
                    "health": comp.health_percentage,
                    "status": comp.status.value,
                    "urgency": comp.urgency.value,
                    "prediction": comp.prediction,
                    "remaining_km": comp.estimated_remaining_km,
                    "remaining_days": comp.estimated_remaining_days,
                    "next_service": comp.next_service,
                    "factors": comp.factors,
                    "confidence": comp.confidence,
                }
                for name, comp in prediction.components.items()
            },
            "urgent_items": prediction.urgent_items,
            "recommendations": prediction.recommendations,
            "next_service_km": prediction.estimated_next_service_km,
            "next_service_date": prediction.estimated_next_service_date,
        }


# Create singleton
maintenance_predictor = MaintenancePredictor()
