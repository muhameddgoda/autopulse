"""
AutoPulse Scoring API - CONSOLIDATED
====================================
All scoring logic in ONE place:
- Rule-based scoring (baseline)
- ML-enhanced scoring (when models are trained)
- Hybrid scoring (rules + ML together)

Endpoints:
- POST /api/scoring/trips/{trip_id}/calculate    - Calculate/recalculate score
- POST /api/scoring/trips/{trip_id}/hybrid       - Get hybrid score (rules + ML)
- POST /api/scoring/backfill/{vehicle_id}        - Backfill scores for trips
- GET  /api/scoring/trips/{vehicle_id}           - Get trips with scores
- GET  /api/scoring/trips/{trip_id}/details      - Get detailed score breakdown
- GET  /api/scoring/trips/{trip_id}/timeline     - Get trip telemetry timeline
- GET  /api/scoring/history/{vehicle_id}         - Get score history for charts
- GET  /api/scoring/summary/{vehicle_id}         - Get driver summary
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.telemetry import Trip, TelemetryReading

router = APIRouter(prefix="/api/scoring", tags=["scoring"])


def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)

# ============================================
# SCORING CONFIGURATION
# ============================================


class ScoringConfig:
    """Centralized scoring thresholds and weights"""
    # Penalty weights
    HARSH_BRAKE_PENALTY = 3.0  # Points per harsh brake
    HARSH_ACCEL_PENALTY = 2.0  # Points per harsh acceleration
    SPEEDING_PENALTY = 0.5  # Points per % of time speeding
    HIGH_SPEED_PENALTY = 0.2  # Points per km/h over 180
    HIGH_SPEED_THRESHOLD = 180  # km/h
    
    # Behavior thresholds
    BEHAVIOR_THRESHOLDS = {
        "exemplary": 85,
        "calm": 70,
        "normal": 55,
        "aggressive": 40,
        # Below 40 = dangerous
    }
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        "low": 70,
        "medium": 50,
        "high": 30,
        # Below 30 = critical
    }

# ============================================
# RESPONSE MODELS
# ============================================


class TripWithScoreResponse(BaseModel):
    """Trip response including driver score"""
    id: str
    vehicle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    is_active: bool
    distance_km: Optional[float]
    duration_seconds: Optional[int]
    avg_speed_kmh: Optional[float]
    max_speed_kmh: Optional[float]
    driver_score: Optional[float]
    behavior_label: Optional[str]
    risk_level: Optional[str]
    harsh_brake_count: Optional[int]
    harsh_accel_count: Optional[int]
    speeding_percentage: Optional[float]
    ml_enhanced: Optional[bool]


class DriverSummaryResponse(BaseModel):
    """Driver summary for aggregate views"""
    total_trips: int
    score_statistics: dict
    overall_behavior: str
    trend: str
    totals: dict
    events_per_100km: dict


class ScoreBackfillResponse(BaseModel):
    """Response for score backfill operation"""
    total_found: int
    scored: int
    failed: int


class HybridScoreResponse(BaseModel):
    """Response for hybrid scoring"""
    trip_id: str
    rule_score: float
    rule_behavior: str
    rule_risk: str
    ml_available: bool
    ml_behavior: Optional[str]
    ml_confidence: Optional[Dict[str, float]]
    ml_is_anomaly: Optional[bool]
    ml_anomaly_score: Optional[float]
    final_score: float
    final_behavior: str
    final_risk: str

# ============================================
# INTERNAL SCORING FUNCTIONS
# ============================================


async def calculate_trip_score_internal(trip_id: str, db: AsyncSession) -> Dict[str, Any]:
    """
    Internal function to calculate trip score using rules.
    Can be imported and called from other modules (e.g., trips.py)
    
    Uses centralized ScoringConfig for consistency.
    """
    # Get trip info
    query = text("""
        SELECT id, vehicle_id, is_active, start_time, end_time,
               distance_km, duration_seconds, avg_speed_kmh, max_speed_kmh
        FROM trips WHERE id = :trip_id
    """)
    result = await db.execute(query, {"trip_id": trip_id})
    trip = result.fetchone()
    
    if not trip:
        raise ValueError(f"Trip {trip_id} not found")
    
    if trip.is_active:
        raise ValueError(f"Cannot score active trip {trip_id}")
    
    # Get telemetry stats
    stats_query = text("""
        SELECT 
            COUNT(*) as reading_count,
            COALESCE(SUM(CASE WHEN is_harsh_braking THEN 1 ELSE 0 END), 0) as harsh_brakes,
            COALESCE(SUM(CASE WHEN is_harsh_acceleration THEN 1 ELSE 0 END), 0) as harsh_accels,
            COALESCE(SUM(CASE WHEN is_speeding THEN 1 ELSE 0 END), 0) as speeding_count
        FROM telemetry_readings
        WHERE vehicle_id = :vehicle_id
          AND time >= :start_time
          AND time <= :end_time
    """)
    
    stats_result = await db.execute(stats_query, {
        "vehicle_id": str(trip.vehicle_id),
        "start_time": trip.start_time,
        "end_time": trip.end_time or utc_now()
    })
    stats = stats_result.fetchone()
    
    # Calculate score using centralized config
    harsh_brakes = stats.harsh_brakes or 0
    harsh_accels = stats.harsh_accels or 0
    speeding_count = stats.speeding_count or 0
    reading_count = stats.reading_count or 1
    
    speeding_percentage = (speeding_count / reading_count) * 100 if reading_count > 0 else 0
    
    # Apply scoring rules
    score = 100.0
    score -= harsh_brakes * ScoringConfig.HARSH_BRAKE_PENALTY
    score -= harsh_accels * ScoringConfig.HARSH_ACCEL_PENALTY
    score -= speeding_percentage * ScoringConfig.SPEEDING_PENALTY
    
    # High speed penalty
    if trip.max_speed_kmh and trip.max_speed_kmh > ScoringConfig.HIGH_SPEED_THRESHOLD:
        score -= (trip.max_speed_kmh - ScoringConfig.HIGH_SPEED_THRESHOLD) * ScoringConfig.HIGH_SPEED_PENALTY
    
    score = max(0, min(100, score))
    
    # Determine behavior
    behavior = "dangerous"
    for label, threshold in ScoringConfig.BEHAVIOR_THRESHOLDS.items():
        if score >= threshold:
            behavior = label
            break
    
    # Determine risk
    risk = "critical"
    for level, threshold in ScoringConfig.RISK_THRESHOLDS.items():
        if score >= threshold:
            risk = level
            break
    
    # Update trip with score
    update_query = text("""
        UPDATE trips SET
            driver_score = :score,
            behavior_label = :behavior,
            risk_level = :risk,
            harsh_brake_count = :harsh_brakes,
            harsh_accel_count = :harsh_accels,
            speeding_percentage = :speeding_pct,
            updated_at = NOW()
        WHERE id = :trip_id
    """)
    
    await db.execute(update_query, {
        "trip_id": trip_id,
        "score": round(score, 1),
        "behavior": behavior,
        "risk": risk,
        "harsh_brakes": harsh_brakes,
        "harsh_accels": harsh_accels,
        "speeding_pct": round(speeding_percentage, 1)
    })
    await db.commit()
    
    return {
        "trip_id": trip_id,
        "driver_score": round(score, 1),
        "behavior_label": behavior,
        "risk_level": risk,
        "harsh_brake_count": harsh_brakes,
        "harsh_accel_count": harsh_accels,
        "speeding_percentage": round(speeding_percentage, 1)
    }


async def calculate_hybrid_score_internal(
    trip_id: str,
    vehicle_id: str,
    db: AsyncSession
) -> Dict[str, Any]:
    """
    Calculate hybrid score combining rules + ML predictions.
    Returns both rule-based and ML-enhanced results.
    """
    # First get rule-based score
    rule_result = await calculate_trip_score_internal(trip_id, db)
    
    result = {
        "trip_id": trip_id,
        "rule_score": rule_result["driver_score"],
        "rule_behavior": rule_result["behavior_label"],
        "rule_risk": rule_result["risk_level"],
        "ml_available": False,
        "ml_behavior": None,
        "ml_confidence": None,
        "ml_is_anomaly": None,
        "ml_anomaly_score": None,
        "final_score": rule_result["driver_score"],
        "final_behavior": rule_result["behavior_label"],
        "final_risk": rule_result["risk_level"]
    }
    
    # Try to enhance with ML
    try:
        from app.ml.models.hybrid_scorer import hybrid_scorer
        from app.ml.features.extractor import FeatureExtractor
        
        # Get trip for time range
        trip_query = text("""
            SELECT start_time, end_time, total_readings
            FROM trips WHERE id = :trip_id
        """)
        trip_result = await db.execute(trip_query, {"trip_id": trip_id})
        trip = trip_result.fetchone()
        
        if not trip:
            return result
        
        # Get telemetry readings
        readings_query = text("""
            SELECT *
            FROM telemetry_readings
            WHERE vehicle_id = :vehicle_id
              AND time >= :start_time
              AND time <= :end_time
            ORDER BY time
        """)
        
        readings_result = await db.execute(readings_query, {
            "vehicle_id": vehicle_id,
            "start_time": trip.start_time,
            "end_time": trip.end_time or utc_now()
        })
        readings_orm = readings_result.fetchall()
        
        if len(readings_orm) < 30:
            return result
        
        # Convert to feature format
        readings = [{
            "timestamp": r.time.isoformat() if r.time else None,
            "speed_kmh": r.speed_kmh,
            "rpm": r.rpm,
            "gear": r.gear,
            "throttle_position": r.throttle_position,
            "engine_temp": r.engine_temp,
            "oil_temp": r.oil_temp,
            "oil_pressure": r.oil_pressure,
            "fuel_level": r.fuel_level,
            "acceleration_g": r.acceleration_g or 0,
            "is_harsh_braking": r.is_harsh_braking or False,
            "is_harsh_acceleration": r.is_harsh_acceleration or False,
            "is_speeding": r.is_speeding or False,
            "is_idling": r.is_idling or False,
            "driving_mode": r.driving_mode,
            "engine_stress_score": r.engine_stress_score or 0,
        } for r in readings_orm]
        
        # Extract features
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_from_readings(readings, trip_id, vehicle_id)
        
        if features is None:
            return result
        
        # Load ML models and get hybrid score (force reload to ensure fresh models)
        hybrid_scorer.load_models_for_vehicle(vehicle_id, force_reload=True)
        hybrid_result = hybrid_scorer.score_trip(features)
        
        # Update result with ML info
        result["ml_available"] = hybrid_result.ml_available
        result["ml_behavior"] = hybrid_result.ml_behavior
        result["ml_confidence"] = hybrid_result.ml_behavior_confidence
        result["ml_is_anomaly"] = hybrid_result.ml_is_anomaly
        result["ml_anomaly_score"] = hybrid_result.ml_anomaly_score
        result["final_score"] = hybrid_result.final_score
        result["final_behavior"] = hybrid_result.final_behavior
        result["final_risk"] = hybrid_result.final_risk
        
        # Always mark as ML-enhanced when ML models are used
        # Use final score (which may blend rule-based and ML)
        update_query = text("""
            UPDATE trips SET
                driver_score = :score,
                behavior_label = :behavior,
                risk_level = :risk,
                ml_enhanced = TRUE,
                updated_at = NOW()
            WHERE id = :trip_id
        """)
        await db.execute(update_query, {
            "trip_id": trip_id,
            "score": round(result["final_score"], 1),
            "behavior": result["final_behavior"],
            "risk": result["final_risk"]
        })
        await db.commit()
        
    except ImportError:
        # ML modules not available
        pass
    except Exception as e:
        print(f"ML scoring error for trip {trip_id}: {e}")
    
    return result

# ============================================
# API ENDPOINTS
# ============================================


@router.post("/trips/{trip_id}/calculate")
async def calculate_trip_score(
    trip_id: str,
    db: AsyncSession=Depends(get_db)
):
    """
    Calculate or recalculate score for a trip.
    Uses hybrid scoring (rules + ML) when ML models are available.
    """
    # Get vehicle_id from trip
    query = text("SELECT vehicle_id FROM trips WHERE id = :trip_id")
    result = await db.execute(query, {"trip_id": trip_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    try:
        # Use hybrid scoring to get ML enhancement
        return await calculate_hybrid_score_internal(trip_id, str(row.vehicle_id), db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trips/{trip_id}/hybrid", response_model=HybridScoreResponse)
async def calculate_hybrid_score(
    trip_id: str,
    db: AsyncSession=Depends(get_db)
):
    """
    Calculate hybrid score (rules + ML) for a trip
    """
    # Get vehicle_id from trip
    query = text("SELECT vehicle_id FROM trips WHERE id = :trip_id")
    result = await db.execute(query, {"trip_id": trip_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    try:
        return await calculate_hybrid_score_internal(trip_id, str(row.vehicle_id), db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/trips/{vehicle_id}", response_model=List[TripWithScoreResponse])
async def get_trips_with_scores(
    vehicle_id: str,
    limit: int=Query(default=50, ge=1, le=200),
    include_active: bool=Query(default=False),
    db: AsyncSession=Depends(get_db)
):
    """
    Get trips with their driver scores
    """
    active_filter = "" if include_active else "AND is_active = FALSE"
    
    query = text(f"""
        SELECT 
            id, vehicle_id, start_time, end_time, is_active,
            distance_km, duration_seconds, avg_speed_kmh, max_speed_kmh,
            driver_score, behavior_label, risk_level,
            harsh_brake_count, harsh_accel_count, speeding_percentage, ml_enhanced
        FROM trips
        WHERE vehicle_id = :vehicle_id
        {active_filter}
        ORDER BY start_time DESC
        LIMIT :limit
    """)
    
    result = await db.execute(query, {"vehicle_id": vehicle_id, "limit": limit})
    trips = result.fetchall()
    
    return [
        TripWithScoreResponse(
            id=str(t.id),
            vehicle_id=str(t.vehicle_id),
            start_time=t.start_time,
            end_time=t.end_time,
            is_active=t.is_active,
            distance_km=t.distance_km,
            duration_seconds=t.duration_seconds,
            avg_speed_kmh=t.avg_speed_kmh,
            max_speed_kmh=t.max_speed_kmh,
            driver_score=t.driver_score,
            behavior_label=t.behavior_label,
            risk_level=t.risk_level,
            harsh_brake_count=t.harsh_brake_count,
            harsh_accel_count=t.harsh_accel_count,
            speeding_percentage=t.speeding_percentage,
            ml_enhanced=t.ml_enhanced
        )
        for t in trips
    ]


@router.get("/trips/{trip_id}/details")
async def get_trip_score_details(
    trip_id: str,
    db: AsyncSession=Depends(get_db)
):
    """
    Get detailed scoring breakdown for a trip
    """
    query = text("""
        SELECT 
            t.*,
            (SELECT COUNT(*) FROM telemetry_readings tr 
             WHERE tr.vehicle_id = t.vehicle_id 
             AND tr.time BETWEEN t.start_time AND COALESCE(t.end_time, NOW())
            ) as reading_count
        FROM trips t
        WHERE t.id = :trip_id
    """)
    
    result = await db.execute(query, {"trip_id": trip_id})
    trip = result.fetchone()
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    # Build response with insights
    response = {
        "trip_id": str(trip.id),
        "vehicle_id": str(trip.vehicle_id),
        "start_time": trip.start_time.isoformat() if trip.start_time else None,
        "end_time": trip.end_time.isoformat() if trip.end_time else None,
        "duration_seconds": trip.duration_seconds,
        "distance_km": trip.distance_km,
        "driver_score": trip.driver_score,
        "behavior_label": trip.behavior_label,
        "risk_level": trip.risk_level,
        "ml_enhanced": getattr(trip, 'ml_enhanced', False),
        "harsh_brake_count": trip.harsh_brake_count or 0,
        "harsh_accel_count": trip.harsh_accel_count or 0,
        "speeding_percentage": trip.speeding_percentage or 0,
        "avg_speed_kmh": trip.avg_speed_kmh,
        "max_speed_kmh": trip.max_speed_kmh,
        "reading_count": trip.reading_count
    }
    
    # Generate insights
    insights = []
    recommendations = []
    
    harsh_brakes = trip.harsh_brake_count or 0
    harsh_accels = trip.harsh_accel_count or 0
    speeding_pct = trip.speeding_percentage or 0
    
    if harsh_brakes == 0:
        insights.append("No harsh braking events - excellent control")
    elif harsh_brakes <= 3:
        insights.append(f"{harsh_brakes} harsh braking events detected")
    else:
        insights.append(f"{harsh_brakes} harsh braking events - needs improvement")
        recommendations.append("Maintain greater following distance to reduce hard braking")
    
    if harsh_accels == 0:
        insights.append("Smooth acceleration patterns")
    elif harsh_accels <= 3:
        insights.append(f"{harsh_accels} rapid acceleration events")
    else:
        insights.append(f"{harsh_accels} aggressive acceleration events")
        recommendations.append("Gentler acceleration improves fuel efficiency and tire life")
    
    if speeding_pct > 10:
        insights.append(f"Speeding {speeding_pct:.1f}% of trip time")
        recommendations.append("Reduce speeding for safer driving")
    elif speeding_pct == 0:
        insights.append("No speeding detected - excellent speed compliance")
    
    if trip.max_speed_kmh:
        insights.append(f"Top speed: {trip.max_speed_kmh:.0f} km/h")
    
    if trip.driver_score and trip.driver_score >= 80 and not recommendations:
        recommendations.append("Keep up the great driving! Your habits are safe and efficient.")
    
    response["insights"] = insights
    response["recommendations"] = recommendations
    
    return response


@router.get("/trips/{trip_id}/timeline")
async def get_trip_timeline(
    trip_id: str,
    db: AsyncSession=Depends(get_db)
):
    """
    Get telemetry timeline for a trip (for behavior chart)
    """
    # Get trip info
    trip_query = text("""
        SELECT id, vehicle_id, start_time, end_time
        FROM trips WHERE id = :trip_id
    """)
    trip_result = await db.execute(trip_query, {"trip_id": trip_id})
    trip = trip_result.fetchone()
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    # Get telemetry readings
    readings_query = text("""
        SELECT 
            time, speed_kmh, rpm, throttle_position,
            is_harsh_braking, is_harsh_acceleration, is_speeding, acceleration_g
        FROM telemetry_readings
        WHERE vehicle_id = :vehicle_id
          AND time >= :start_time
          AND time <= :end_time
        ORDER BY time
    """)
    
    readings_result = await db.execute(readings_query, {
        "vehicle_id": str(trip.vehicle_id),
        "start_time": trip.start_time,
        "end_time": trip.end_time or utc_now()
    })
    readings = readings_result.fetchall()
    
    if not readings:
        return {"trip_id": trip_id, "timeline": [], "count": 0}
    
    # Calculate running score at each point
    timeline = []
    harsh_brakes_so_far = 0
    harsh_accels_so_far = 0
    speeding_count = 0
    start_time = trip.start_time
    
    for i, r in enumerate(readings):
        if r.is_harsh_braking:
            harsh_brakes_so_far += 1
        if r.is_harsh_acceleration:
            harsh_accels_so_far += 1
        if r.is_speeding:
            speeding_count += 1
        
        readings_so_far = i + 1
        speeding_pct = (speeding_count / readings_so_far) * 100
        
        score = 100.0
        score -= harsh_brakes_so_far * ScoringConfig.HARSH_BRAKE_PENALTY
        score -= harsh_accels_so_far * ScoringConfig.HARSH_ACCEL_PENALTY
        score -= speeding_pct * ScoringConfig.SPEEDING_PENALTY
        score = max(0, min(100, score))
        
        time_offset = (r.time - start_time).total_seconds()
        
        timeline.append({
            "time_offset": round(time_offset),
            "time": r.time.isoformat(),
            "speed_kmh": r.speed_kmh,
            "rpm": r.rpm,
            "score": round(score, 1),
            "is_harsh_brake": r.is_harsh_braking or False,
            "is_harsh_accel": r.is_harsh_acceleration or False,
            "is_speeding": r.is_speeding or False,
            "acceleration_g": round(r.acceleration_g, 2) if r.acceleration_g else 0
        })
    
    return {"trip_id": trip_id, "timeline": timeline, "count": len(timeline)}


@router.get("/history/{vehicle_id}")
async def get_score_history(
    vehicle_id: str,
    limit: int=Query(default=10, ge=1, le=50),
    days: int=Query(default=None, ge=1, le=365),
    db: AsyncSession=Depends(get_db)
):
    """
    Get score history for charts (score over time)
    """
    if days:
        cutoff = utc_now() - timedelta(days=days)
        query = text("""
            SELECT id, start_time, driver_score, behavior_label, distance_km
            FROM trips
            WHERE vehicle_id = :vehicle_id
              AND is_active = FALSE
              AND driver_score IS NOT NULL
              AND start_time >= :cutoff
            ORDER BY start_time DESC
            LIMIT :limit
        """)
        result = await db.execute(query, {"vehicle_id": vehicle_id, "limit": limit, "cutoff": cutoff})
    else:
        query = text("""
            SELECT id, start_time, driver_score, behavior_label, distance_km
            FROM trips
            WHERE vehicle_id = :vehicle_id
              AND is_active = FALSE
              AND driver_score IS NOT NULL
            ORDER BY start_time DESC
            LIMIT :limit
        """)
        result = await db.execute(query, {"vehicle_id": vehicle_id, "limit": limit})
    
    trips = result.fetchall()
    
    # Return in chronological order for charts
    history = [
        {
            "trip_id": str(t.id),
            "date": t.start_time.isoformat(),
            "score": t.driver_score,
            "behavior": t.behavior_label,
            "distance_km": t.distance_km
        }
        for t in reversed(trips)
    ]
    
    return {"vehicle_id": vehicle_id, "history": history, "count": len(history)}


@router.get("/summary/{vehicle_id}", response_model=DriverSummaryResponse)
async def get_driver_summary(
    vehicle_id: str,
    days: int=Query(default=7, ge=1, le=365),
    db: AsyncSession=Depends(get_db)
):
    """
    Get aggregated driver behavior summary
    """
    cutoff = utc_now() - timedelta(days=days)
    
    # Get trip statistics
    query = text("""
        SELECT 
            COUNT(*) as trip_count,
            COALESCE(AVG(driver_score), 0) as avg_score,
            COALESCE(MIN(driver_score), 0) as min_score,
            COALESCE(MAX(driver_score), 0) as max_score,
            COALESCE(SUM(distance_km), 0) as total_distance,
            COALESCE(SUM(duration_seconds), 0) as total_duration,
            COALESCE(SUM(harsh_brake_count), 0) as total_brakes,
            COALESCE(SUM(harsh_accel_count), 0) as total_accels
        FROM trips
        WHERE vehicle_id = :vehicle_id
          AND is_active = FALSE
          AND start_time >= :cutoff
    """)
    
    result = await db.execute(query, {"vehicle_id": vehicle_id, "cutoff": cutoff})
    stats = result.fetchone()
    
    if not stats or stats.trip_count == 0:
        return DriverSummaryResponse(
            total_trips=0,
            score_statistics={"average": 0, "minimum": 0, "maximum": 0},
            overall_behavior="unknown",
            trend="stable",
            totals={"distance_km": 0, "duration_hours": 0, "harsh_brakes": 0, "harsh_accelerations": 0},
            events_per_100km={"harsh_brakes": 0, "harsh_accels": 0}
        )
    
    # Determine behavior using centralized thresholds
    avg_score = stats.avg_score or 0
    behavior = "dangerous"
    for label, threshold in ScoringConfig.BEHAVIOR_THRESHOLDS.items():
        if avg_score >= threshold:
            behavior = label
            break
    
    # Calculate trend
    trend = "stable"
    if stats.trip_count >= 4:
        mid_cutoff = cutoff + timedelta(days=days / 2)
        
        first_half_query = text("""
            SELECT COALESCE(AVG(driver_score), 0) as avg_score
            FROM trips
            WHERE vehicle_id = :vehicle_id AND is_active = FALSE
              AND start_time >= :cutoff AND start_time < :mid_cutoff
        """)
        second_half_query = text("""
            SELECT COALESCE(AVG(driver_score), 0) as avg_score
            FROM trips
            WHERE vehicle_id = :vehicle_id AND is_active = FALSE
              AND start_time >= :mid_cutoff
        """)
        
        first_result = await db.execute(first_half_query, {"vehicle_id": vehicle_id, "cutoff": cutoff, "mid_cutoff": mid_cutoff})
        second_result = await db.execute(second_half_query, {"vehicle_id": vehicle_id, "mid_cutoff": mid_cutoff})
        
        first_avg = first_result.fetchone().avg_score or 0
        second_avg = second_result.fetchone().avg_score or 0
        
        if second_avg > first_avg + 3:
            trend = "improving"
        elif second_avg < first_avg - 3:
            trend = "declining"
    
    # Events per 100km
    total_distance = stats.total_distance or 0
    brakes_per_100km = (stats.total_brakes / (total_distance / 100)) if total_distance > 0 else 0
    accels_per_100km = (stats.total_accels / (total_distance / 100)) if total_distance > 0 else 0
    
    return DriverSummaryResponse(
        total_trips=stats.trip_count,
        score_statistics={
            "average": round(avg_score, 1),
            "minimum": round(stats.min_score or 0, 1),
            "maximum": round(stats.max_score or 0, 1),
        },
        overall_behavior=behavior,
        trend=trend,
        totals={
            "distance_km": round(total_distance, 1),
            "duration_hours": round((stats.total_duration or 0) / 3600, 1),
            "harsh_brakes": stats.total_brakes or 0,
            "harsh_accelerations": stats.total_accels or 0,
        },
        events_per_100km={
            "harsh_brakes": round(brakes_per_100km, 1),
            "harsh_accels": round(accels_per_100km, 1),
        }
    )


@router.post("/backfill/{vehicle_id}", response_model=ScoreBackfillResponse)
async def backfill_trip_scores(
    vehicle_id: str,
    use_hybrid: bool=Query(default=False, description="Use hybrid scoring if ML available"),
    limit: int=Query(default=100, ge=1, le=500),
    db: AsyncSession=Depends(get_db)
):
    """
    Backfill scores for existing trips that don't have scores
    """
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
        try:
            if use_hybrid:
                await calculate_hybrid_score_internal(str(trip.id), vehicle_id, db)
            else:
                await calculate_trip_score_internal(str(trip.id), db)
            scored += 1
        except Exception as e:
            print(f"Failed to score trip {trip.id}: {e}")
            failed += 1
    
    return ScoreBackfillResponse(total_found=len(trips), scored=scored, failed=failed)
