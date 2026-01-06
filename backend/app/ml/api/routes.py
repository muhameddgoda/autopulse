"""
AutoPulse ML API Routes
Endpoints for driver scoring and behavior analysis
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta

from app.ml.features.extractor import FeatureExtractor, TripFeatures
from app.ml.models.driver_scorer import DriverScorer, DriverScore


class MLService:
    """
    ML Service for driver behavior analysis
    Provides scoring and analysis functions for the backend API
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scorer = DriverScorer()
    
    def analyze_trip(
        self,
        readings: List[Dict],
        trip_id: str = "",
        vehicle_id: str = ""
    ) -> Optional[Dict]:
        """
        Analyze a completed trip and return driver score
        
        Args:
            readings: List of telemetry readings for the trip
            trip_id: Trip identifier
            vehicle_id: Vehicle identifier
            
        Returns:
            Dictionary with score, insights, and recommendations
        """
        # Extract features
        features = self.feature_extractor.extract_from_readings(
            readings, trip_id, vehicle_id
        )
        
        if features is None:
            return {
                "error": "Insufficient data for analysis",
                "min_readings_required": 30,
                "readings_provided": len(readings) if readings else 0
            }
        
        # Calculate score
        score = self.scorer.score_trip(features)
        
        # Convert to dict for JSON response
        result = self.scorer.score_to_dict(score)
        
        # Add feature details
        result["features"] = self.feature_extractor.features_to_dict(features)
        
        return result
    
    def analyze_readings_batch(
        self,
        readings: List[Dict],
        window_seconds: int = 300
    ) -> List[Dict]:
        """
        Analyze telemetry in batches/windows
        Useful for long trips or continuous monitoring
        
        Args:
            readings: All telemetry readings
            window_seconds: Size of each analysis window
            
        Returns:
            List of analysis results for each window
        """
        if not readings:
            return []
        
        results = []
        current_window = []
        window_start = None
        
        for reading in readings:
            timestamp = reading.get('timestamp') or reading.get('time')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            if window_start is None:
                window_start = timestamp
            
            # Check if we've exceeded window size
            if timestamp and window_start:
                elapsed = (timestamp - window_start).total_seconds()
                
                if elapsed >= window_seconds and len(current_window) >= 30:
                    # Analyze current window
                    analysis = self.analyze_trip(current_window)
                    if analysis and "error" not in analysis:
                        analysis["window_start"] = window_start.isoformat()
                        analysis["window_end"] = timestamp.isoformat()
                        results.append(analysis)
                    
                    # Start new window
                    current_window = []
                    window_start = timestamp
            
            current_window.append(reading)
        
        # Analyze remaining readings
        if len(current_window) >= 30:
            analysis = self.analyze_trip(current_window)
            if analysis and "error" not in analysis:
                results.append(analysis)
        
        return results
    
    def get_driver_summary(
        self,
        trip_scores: List[Dict]
    ) -> Dict:
        """
        Generate overall driver summary from multiple trips
        
        Args:
            trip_scores: List of individual trip scores
            
        Returns:
            Aggregated driver statistics and trends
        """
        if not trip_scores:
            return {"error": "No trip data available"}
        
        scores = [t.get("score", 0) for t in trip_scores if "score" in t]
        
        if not scores:
            return {"error": "No valid scores found"}
        
        # Calculate statistics
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Count behavior labels
        behaviors = [t.get("behavior", "unknown") for t in trip_scores]
        behavior_counts = {}
        for b in behaviors:
            behavior_counts[b] = behavior_counts.get(b, 0) + 1
        
        # Aggregate harsh events
        total_harsh_brakes = sum(
            t.get("features", {}).get("events", {}).get("harsh_brake_count", 0)
            for t in trip_scores
        )
        total_harsh_accels = sum(
            t.get("features", {}).get("events", {}).get("harsh_accel_count", 0)
            for t in trip_scores
        )
        
        # Total distance and time
        total_distance = sum(
            t.get("trip_info", {}).get("distance_km", 0)
            for t in trip_scores
        )
        total_duration = sum(
            t.get("trip_info", {}).get("duration_seconds", 0)
            for t in trip_scores
        )
        
        # Determine overall behavior
        if avg_score >= 90:
            overall_behavior = "exemplary"
        elif avg_score >= 75:
            overall_behavior = "calm"
        elif avg_score >= 60:
            overall_behavior = "normal"
        elif avg_score >= 40:
            overall_behavior = "aggressive"
        else:
            overall_behavior = "dangerous"
        
        # Calculate trend (if enough trips)
        trend = "stable"
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:-3]) / max(1, len(scores) - 3) if len(scores) > 3 else scores[0]
            
            if recent_avg > older_avg + 5:
                trend = "improving"
            elif recent_avg < older_avg - 5:
                trend = "declining"
        
        return {
            "total_trips": len(trip_scores),
            "score_statistics": {
                "average": round(avg_score, 1),
                "minimum": round(min_score, 1),
                "maximum": round(max_score, 1),
            },
            "overall_behavior": overall_behavior,
            "trend": trend,
            "behavior_distribution": behavior_counts,
            "totals": {
                "distance_km": round(total_distance, 2),
                "duration_hours": round(total_duration / 3600, 2),
                "harsh_brakes": total_harsh_brakes,
                "harsh_accelerations": total_harsh_accels,
            },
            "events_per_100km": {
                "harsh_brakes": round(total_harsh_brakes / max(0.01, total_distance / 100), 1),
                "harsh_accels": round(total_harsh_accels / max(0.01, total_distance / 100), 1),
            }
        }
    
    def compare_trips(
        self,
        trip1: Dict,
        trip2: Dict
    ) -> Dict:
        """
        Compare two trips and highlight differences
        
        Args:
            trip1: First trip analysis
            trip2: Second trip analysis
            
        Returns:
            Comparison with improvements/declines
        """
        score1 = trip1.get("score", 0)
        score2 = trip2.get("score", 0)
        
        score_diff = score2 - score1
        
        comparison = {
            "score_change": round(score_diff, 1),
            "improved": score_diff > 0,
            "trip1_score": score1,
            "trip2_score": score2,
            "component_changes": {},
        }
        
        # Compare components
        comp1 = trip1.get("components", {})
        comp2 = trip2.get("components", {})
        
        for name in comp1.keys():
            if name in comp2:
                diff = comp2[name].get("score", 0) - comp1[name].get("score", 0)
                comparison["component_changes"][name] = {
                    "change": round(diff, 1),
                    "improved": diff > 0
                }
        
        return comparison


# Create singleton instance
ml_service = MLService()
