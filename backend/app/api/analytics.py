"""
Analytics endpoints
Handles stats, summaries, and data exports
"""
from typing import Optional
from datetime import datetime, timedelta, timezone
from uuid import UUID
from io import StringIO
import csv

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, text

from app.database import get_db
from app.models.telemetry import Vehicle, TelemetryReading, Trip

router = APIRouter()


def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)

# ============================================
# STATS ENDPOINTS
# ============================================


@router.get("/stats/weekly/{vehicle_id}")
async def get_weekly_stats(
    vehicle_id: UUID,
    db: AsyncSession=Depends(get_db)
):
    """
    Get aggregated weekly statistics for a vehicle
    """
    week_ago = utc_now() - timedelta(days=7)
    
    # Use raw SQL to avoid SQLAlchemy cast issues
    query = text("""
        SELECT 
            COUNT(*) as total_readings,
            COALESCE(AVG(speed_kmh), 0) as avg_speed,
            COALESCE(MAX(speed_kmh), 0) as max_speed,
            COALESCE(AVG(rpm), 0) as avg_rpm,
            COALESCE(SUM(CASE WHEN is_harsh_braking THEN 1 ELSE 0 END), 0) as harsh_braking_events,
            COALESCE(SUM(CASE WHEN is_speeding THEN 1 ELSE 0 END), 0) as speeding_events
        FROM telemetry_readings
        WHERE vehicle_id = :vehicle_id
          AND time >= :since
    """)
    
    result = await db.execute(query, {
        "vehicle_id": str(vehicle_id),
        "since": week_ago
    })
    stats = result.first()
    
    return {
        "vehicle_id": str(vehicle_id),
        "period": "7_days",
        "total_readings": stats.total_readings or 0,
        "avg_speed_kmh": round(float(stats.avg_speed or 0), 2),
        "max_speed_kmh": round(float(stats.max_speed or 0), 2),
        "avg_rpm": round(float(stats.avg_rpm or 0), 0),
        "harsh_braking_events": int(stats.harsh_braking_events or 0),
        "speeding_events": int(stats.speeding_events or 0)
    }


@router.get("/stats/daily/{vehicle_id}")
async def get_daily_stats(
    vehicle_id: UUID,
    days: int=Query(default=7, ge=1, le=90),
    db: AsyncSession=Depends(get_db)
):
    """
    Get daily breakdown statistics for a vehicle
    """
    since = utc_now() - timedelta(days=days)
    
    # Use raw SQL for date truncation (works with TimescaleDB)
    query = text("""
        SELECT 
            DATE(time AT TIME ZONE 'UTC') as day,
            COUNT(*) as readings,
            AVG(speed_kmh) as avg_speed,
            MAX(speed_kmh) as max_speed,
            SUM(CASE WHEN is_harsh_braking THEN 1 ELSE 0 END) as harsh_brakes,
            SUM(CASE WHEN is_speeding THEN 1 ELSE 0 END) as speeding
        FROM telemetry_readings
        WHERE vehicle_id = :vehicle_id
          AND time >= :since
        GROUP BY DATE(time AT TIME ZONE 'UTC')
        ORDER BY day DESC
    """)
    
    result = await db.execute(query, {
        "vehicle_id": str(vehicle_id),
        "since": since
    })
    
    rows = result.fetchall()
    
    return {
        "vehicle_id": str(vehicle_id),
        "days_requested": days,
        "daily_stats": [
            {
                "date": str(row.day),
                "readings": row.readings,
                "avg_speed_kmh": round(float(row.avg_speed or 0), 2),
                "max_speed_kmh": round(float(row.max_speed or 0), 2),
                "harsh_brakes": int(row.harsh_brakes or 0),
                "speeding_events": int(row.speeding or 0)
            }
            for row in rows
        ]
    }

# ============================================
# CSV EXPORT ENDPOINTS
# ============================================


@router.get("/export/telemetry/{vehicle_id}")
async def export_telemetry_csv(
    vehicle_id: UUID,
    hours: int=Query(default=24, ge=1, le=168),
    db: AsyncSession=Depends(get_db)
):
    """
    Export telemetry data as CSV for analysis
    """
    since = utc_now() - timedelta(hours=hours)
    
    result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
        .where(TelemetryReading.time >= since)
        .order_by(TelemetryReading.time)
    )
    readings = result.scalars().all()
    
    if not readings:
        raise HTTPException(status_code=404, detail="No telemetry data found")
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "timestamp", "speed_kmh", "rpm", "gear", "throttle_position",
        "engine_temp", "oil_temp", "oil_pressure", "fuel_level",
        "acceleration_g", "is_harsh_braking", "is_harsh_acceleration",
        "is_speeding", "is_idling", "driving_mode", "engine_stress_score"
    ])
    
    # Data rows
    for r in readings:
        writer.writerow([
            r.time.isoformat() if r.time else "",
            r.speed_kmh,
            r.rpm,
            r.gear,
            r.throttle_position,
            r.engine_temp,
            r.oil_temp,
            r.oil_pressure,
            r.fuel_level,
            r.acceleration_g,
            r.is_harsh_braking,
            r.is_harsh_acceleration,
            r.is_speeding,
            r.is_idling,
            r.driving_mode,
            r.engine_stress_score
        ])
    
    output.seek(0)
    
    filename = f"telemetry_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/export/trips/{vehicle_id}")
async def export_trips_csv(
    vehicle_id: UUID,
    limit: int=Query(default=100, ge=1, le=1000),
    db: AsyncSession=Depends(get_db)
):
    """
    Export trips data as CSV
    """
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == False)
        .order_by(Trip.start_time.desc())
        .limit(limit)
    )
    trips = result.scalars().all()
    
    if not trips:
        raise HTTPException(status_code=404, detail="No trips found")
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "trip_id", "start_time", "end_time", "duration_seconds",
        "distance_km", "avg_speed_kmh", "max_speed_kmh",
        "fuel_consumed_liters", "driver_score", "behavior_label",
        "risk_level", "harsh_brake_count", "harsh_accel_count",
        "speeding_percentage", "total_readings"
    ])
    
    # Data rows
    for t in trips:
        writer.writerow([
            str(t.id),
            t.start_time.isoformat() if t.start_time else "",
            t.end_time.isoformat() if t.end_time else "",
            t.duration_seconds,
            t.distance_km,
            t.avg_speed_kmh,
            t.max_speed_kmh,
            t.fuel_consumed_liters,
            t.driver_score,
            t.behavior_label,
            t.risk_level,
            t.harsh_brake_count,
            t.harsh_accel_count,
            t.speeding_percentage,
            t.total_readings
        ])
    
    output.seek(0)
    
    filename = f"trips_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ============================================
# SUMMARY ENDPOINTS
# ============================================


@router.get("/summary/{vehicle_id}")
async def get_vehicle_summary(
    vehicle_id: UUID,
    days: int=Query(default=30, ge=1, le=365),
    db: AsyncSession=Depends(get_db)
):
    """
    Get comprehensive summary of vehicle performance over a period
    """
    since = utc_now() - timedelta(days=days)
    
    # Get trip statistics
    trip_query = text("""
        SELECT 
            COUNT(*) as total_trips,
            SUM(distance_km) as total_distance,
            SUM(duration_seconds) as total_duration,
            AVG(driver_score) as avg_score,
            SUM(fuel_consumed_liters) as total_fuel,
            AVG(avg_speed_kmh) as overall_avg_speed,
            MAX(max_speed_kmh) as top_speed
        FROM trips
        WHERE vehicle_id = :vehicle_id
          AND is_active = false
          AND start_time >= :since
    """)
    
    result = await db.execute(trip_query, {
        "vehicle_id": str(vehicle_id),
        "since": since
    })
    trip_stats = result.fetchone()
    
    # Get behavior breakdown
    behavior_query = text("""
        SELECT 
            behavior_label,
            COUNT(*) as count
        FROM trips
        WHERE vehicle_id = :vehicle_id
          AND is_active = false
          AND start_time >= :since
          AND behavior_label IS NOT NULL
        GROUP BY behavior_label
    """)
    
    behavior_result = await db.execute(behavior_query, {
        "vehicle_id": str(vehicle_id),
        "since": since
    })
    behaviors = {row.behavior_label: row.count for row in behavior_result.fetchall()}
    
    return {
        "vehicle_id": str(vehicle_id),
        "period_days": days,
        "total_trips": trip_stats.total_trips or 0,
        "total_distance_km": round(float(trip_stats.total_distance or 0), 2),
        "total_duration_hours": round(float(trip_stats.total_duration or 0) / 3600, 2),
        "average_score": round(float(trip_stats.avg_score or 0), 1),
        "total_fuel_liters": round(float(trip_stats.total_fuel or 0), 2),
        "overall_avg_speed_kmh": round(float(trip_stats.overall_avg_speed or 0), 2),
        "top_speed_kmh": round(float(trip_stats.top_speed or 0), 2),
        "behavior_breakdown": behaviors
    }
