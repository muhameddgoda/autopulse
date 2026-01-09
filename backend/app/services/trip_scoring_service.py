"""
AutoPulse Trip Scoring Service
Automatically scores trips when they end and persists results to database

This service:
1. Extracts features from telemetry readings for a trip
2. Calculates driver behavior score (rule-based + ML if available)
3. Persists score to the trips table
4. Can backfill scores for existing trips
"""

from typing import Optional, Dict, Tuple
from datetime import datetime
from uuid import UUID
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession
import logging

# Import scoring components
from app.ml.features.extractor import FeatureExtractor, TripFeatures
from app.ml.models.driver_scorer import DriverScorer, DriverScore
from app.ml.models.hybrid_scorer import HybridScorer, HybridScore

logger = logging.getLogger(__name__)


class TripScoringService:
    """
    Service to calculate and persist driver behavior scores for trips
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.rule_scorer = DriverScorer()
        self.hybrid_scorer = HybridScorer()
    
    async def score_trip_on_end(
        self,
        db: AsyncSession,
        trip_id: str,
        vehicle_id: str,
        use_ml: bool = True
    ) -> Optional[Dict]:
        """
        Score a trip when it ends and persist to database
        
        Args:
            db: Database session
            trip_id: UUID of the trip
            vehicle_id: UUID of the vehicle
            use_ml: Whether to use ML enhancement if available
            
        Returns:
            Score result dict or None if scoring failed
        """
        try:
            # 1. Extract features from telemetry readings
            features = await self._extract_trip_features(db, trip_id, vehicle_id)
            
            if not features:
                logger.warning(f"No telemetry data for trip {trip_id}, using fallback scoring")
                return await self._fallback_score(db, trip_id)
            
            # 2. Calculate score
            if use_ml:
                # Try hybrid scoring (rules + ML)
                self.hybrid_scorer.load_models_for_vehicle(vehicle_id)
                hybrid_result = self.hybrid_scorer.score_trip(features, use_ml=True)
                
                score = hybrid_result.final_score
                behavior = hybrid_result.final_behavior
                risk = hybrid_result.final_risk
                ml_enhanced = hybrid_result.using_ml
            else:
                # Rule-based only
                rule_result = self.rule_scorer.score_trip(features)
                
                score = rule_result.total_score
                behavior = rule_result.behavior_label.value
                risk = rule_result.risk_level.value
                ml_enhanced = False
            
            # 3. Persist to database
            await self._persist_score(
                db=db,
                trip_id=trip_id,
                driver_score=score,
                behavior_label=behavior,
                risk_level=risk,
                harsh_brake_count=features.harsh_brake_count,
                harsh_accel_count=features.harsh_accel_count,
                speeding_percentage=features.speeding_percentage,
                ml_enhanced=ml_enhanced
            )
            
            logger.info(f"Trip {trip_id} scored: {score:.1f} ({behavior})")
            
            return {
                "trip_id": trip_id,
                "score": round(score, 1),
                "behavior": behavior,
                "risk_level": risk,
                "harsh_brake_count": features.harsh_brake_count,
                "harsh_accel_count": features.harsh_accel_count,
                "speeding_percentage": round(features.speeding_percentage, 1),
                "ml_enhanced": ml_enhanced
            }
            
        except Exception as e:
            logger.error(f"Failed to score trip {trip_id}: {e}")
            return None
    
    async def _extract_trip_features(
        self,
        db: AsyncSession,
        trip_id: str,
        vehicle_id: str
    ) -> Optional[TripFeatures]:
        """
        Extract features from telemetry readings for a trip
        """
        # Get trip time bounds
        trip_query = text("""
            SELECT start_time, end_time, distance_km, duration_seconds
            FROM trips 
            WHERE id = :trip_id
        """)
        trip_result = await db.execute(trip_query, {"trip_id": trip_id})
        trip = trip_result.fetchone()
        
        if not trip:
            return None
        
        start_time = trip.start_time
        end_time = trip.end_time or datetime.utcnow()
        
        # Get telemetry readings for the trip
        readings_query = text("""
            SELECT 
                time,
                speed_kmh,
                rpm,
                throttle_position,
                engine_temp,
                acceleration_ms2,
                acceleration_g,
                is_harsh_braking,
                is_harsh_acceleration,
                is_speeding,
                is_idling,
                is_over_rpm,
                engine_stress_score,
                latitude,
                longitude
            FROM telemetry_readings
            WHERE vehicle_id = :vehicle_id
              AND time >= :start_time
              AND time <= :end_time
            ORDER BY time ASC
        """)
        
        result = await db.execute(readings_query, {
            "vehicle_id": vehicle_id,
            "start_time": start_time,
            "end_time": end_time
        })
        readings = result.fetchall()
        
        if not readings or len(readings) < 2:
            return None
        
        # Convert to feature extractor format
        readings_data = []
        for r in readings:
            readings_data.append({
                "time": r.time,
                "speed_kmh": r.speed_kmh or 0,
                "rpm": r.rpm or 0,
                "throttle_position": r.throttle_position or 0,
                "engine_temp": r.engine_temp or 90,
                "acceleration_ms2": r.acceleration_ms2 or 0,
                "acceleration_g": r.acceleration_g or 0,
                "is_harsh_braking": r.is_harsh_braking or False,
                "is_harsh_acceleration": r.is_harsh_acceleration or False,
                "is_speeding": r.is_speeding or False,
                "is_idling": r.is_idling or False,
                "is_over_rpm": r.is_over_rpm or False,
                "engine_stress_score": r.engine_stress_score or 0,
                "latitude": r.latitude,
                "longitude": r.longitude,
            })
        
        # Extract features
        features = self.feature_extractor.extract_from_readings(
            readings=readings_data,
            trip_id=trip_id,
            vehicle_id=vehicle_id
        )
        
        # Override with trip-level stats if available
        if trip.distance_km:
            features.distance_km = trip.distance_km
        if trip.duration_seconds:
            features.duration_seconds = trip.duration_seconds
        
        return features
    
    async def _persist_score(
        self,
        db: AsyncSession,
        trip_id: str,
        driver_score: float,
        behavior_label: str,
        risk_level: str,
        harsh_brake_count: int,
        harsh_accel_count: int,
        speeding_percentage: float,
        ml_enhanced: bool
    ):
        """
        Persist scoring results to the trips table
        """
        update_query = text("""
            UPDATE trips SET
                driver_score = :driver_score,
                behavior_label = :behavior_label,
                risk_level = :risk_level,
                harsh_brake_count = :harsh_brake_count,
                harsh_accel_count = :harsh_accel_count,
                speeding_percentage = :speeding_percentage,
                ml_enhanced = :ml_enhanced,
                updated_at = NOW()
            WHERE id = :trip_id
        """)
        
        await db.execute(update_query, {
            "trip_id": trip_id,
            "driver_score": driver_score,
            "behavior_label": behavior_label,
            "risk_level": risk_level,
            "harsh_brake_count": harsh_brake_count,
            "harsh_accel_count": harsh_accel_count,
            "speeding_percentage": speeding_percentage,
            "ml_enhanced": ml_enhanced
        })
        await db.commit()
    
    async def _fallback_score(self, db: AsyncSession, trip_id: str) -> Optional[Dict]:
        """
        Fallback scoring when no telemetry data is available
        Uses trip-level statistics to estimate a score
        """
        query = text("""
            SELECT distance_km, duration_seconds, avg_speed_kmh, max_speed_kmh
            FROM trips WHERE id = :trip_id
        """)
        result = await db.execute(query, {"trip_id": trip_id})
        trip = result.fetchone()
        
        if not trip:
            return None
        
        # Simple heuristic scoring
        score = 75.0  # Base score
        
        # Penalize high max speed
        if trip.max_speed_kmh and trip.max_speed_kmh > 160:
            score -= min(20, (trip.max_speed_kmh - 160) / 5)
        
        # Determine behavior
        if score >= 80:
            behavior = "calm"
        elif score >= 60:
            behavior = "normal"
        else:
            behavior = "aggressive"
        
        risk = "low" if score >= 70 else "medium" if score >= 50 else "high"
        
        # Persist
        await self._persist_score(
            db=db,
            trip_id=trip_id,
            driver_score=score,
            behavior_label=behavior,
            risk_level=risk,
            harsh_brake_count=0,
            harsh_accel_count=0,
            speeding_percentage=0,
            ml_enhanced=False
        )
        
        return {
            "trip_id": trip_id,
            "score": round(score, 1),
            "behavior": behavior,
            "risk_level": risk,
            "harsh_brake_count": 0,
            "harsh_accel_count": 0,
            "speeding_percentage": 0,
            "ml_enhanced": False,
            "fallback": True
        }
    
    async def backfill_scores(
        self,
        db: AsyncSession,
        vehicle_id: str,
        limit: int = 100
    ) -> Dict:
        """
        Backfill scores for existing trips that don't have scores
        
        Args:
            db: Database session
            vehicle_id: Vehicle UUID
            limit: Maximum trips to process
            
        Returns:
            Summary of backfill operation
        """
        # Find trips without scores
        query = text("""
            SELECT id FROM trips
            WHERE vehicle_id = :vehicle_id
              AND is_active = FALSE
              AND driver_score IS NULL
            ORDER BY start_time DESC
            LIMIT :limit
        """)
        
        result = await db.execute(query, {"vehicle_id": vehicle_id, "limit": limit})
        trips = result.fetchall()
        
        scored = 0
        failed = 0
        
        for trip in trips:
            result = await self.score_trip_on_end(
                db=db,
                trip_id=str(trip.id),
                vehicle_id=vehicle_id,
                use_ml=True
            )
            if result:
                scored += 1
            else:
                failed += 1
        
        return {
            "total_found": len(trips),
            "scored": scored,
            "failed": failed
        }


# Create singleton instance
trip_scoring_service = TripScoringService()
