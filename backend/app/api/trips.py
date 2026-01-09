"""
Trip management endpoints
Handles trip lifecycle (start, end, list)
"""
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.database import get_db
from app.models.telemetry import Vehicle, Trip
from app.schemas.telemetry import TripResponse

# Import the scoring function for trip end (hybrid for ML enhancement)
from app.api.scoring import calculate_hybrid_score_internal

router = APIRouter()


def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)


@router.post("/start/{vehicle_id}", response_model=TripResponse)
async def start_trip(
    vehicle_id: UUID,
    db: AsyncSession=Depends(get_db)
):
    """
    Start a new trip for a vehicle.
    If there's already an active trip, it will be ended first.
    """
    # Check vehicle exists
    vehicle_result = await db.execute(
        select(Vehicle).where(Vehicle.id == vehicle_id)
    )
    vehicle = vehicle_result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Check for existing active trip
    active_result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == True)
    )
    active_trip = active_result.scalar_one_or_none()
    
    if active_trip:
        # End the existing trip first
        active_trip.is_active = False
        active_trip.end_time = utc_now()
        
        if active_trip.start_time:
            delta = active_trip.end_time - active_trip.start_time
            active_trip.duration_seconds = int(delta.total_seconds())
    
    # Create new trip
    new_trip = Trip(
        vehicle_id=vehicle_id,
        start_time=utc_now(),
        is_active=True,
        fuel_start=None,  # Will be updated with first reading
        total_readings=0
    )
    db.add(new_trip)
    await db.commit()
    await db.refresh(new_trip)
    
    return new_trip


@router.post("/end/{trip_id}", response_model=TripResponse)
async def end_trip(
    trip_id: UUID,
    db: AsyncSession=Depends(get_db)
):
    """
    End an active trip and calculate its score.
    """
    # Get the trip
    result = await db.execute(
        select(Trip).where(Trip.id == trip_id)
    )
    trip = result.scalar_one_or_none()
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    if not trip.is_active:
        raise HTTPException(status_code=400, detail="Trip is not active")
    
    # End the trip
    trip.is_active = False
    trip.end_time = utc_now()
    
    if trip.start_time:
        delta = trip.end_time - trip.start_time
        trip.duration_seconds = int(delta.total_seconds())
    
    # Calculate averages from running totals
    if trip.total_readings and trip.total_readings > 0:
        if trip.speed_sum:
            trip.avg_speed_kmh = trip.speed_sum / trip.total_readings
        if trip.rpm_sum:
            trip.avg_rpm = int(trip.rpm_sum / trip.total_readings)
    
    await db.commit()
    await db.refresh(trip)
    
    # Calculate score automatically on trip end (uses hybrid scoring with ML)
    try:
        await calculate_hybrid_score_internal(str(trip.id), str(trip.vehicle_id), db)
        await db.refresh(trip)
    except Exception as e:
        # Log error but don't fail the endpoint
        print(f"Warning: Score calculation failed for trip {trip_id}: {e}")
    
    return trip


@router.get("/{vehicle_id}", response_model=List[TripResponse])
async def get_trips(
    vehicle_id: UUID,
    limit: int=10,
    include_active: bool=False,
    db: AsyncSession=Depends(get_db)
):
    """
    Get trips for a vehicle.
    By default returns only completed trips, ordered by most recent.
    """
    query = select(Trip).where(Trip.vehicle_id == vehicle_id)
    
    if not include_active:
        query = query.where(Trip.is_active == False)
    
    query = query.order_by(Trip.start_time.desc()).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/active/{vehicle_id}", response_model=Optional[TripResponse])
async def get_active_trip(
    vehicle_id: UUID,
    db: AsyncSession=Depends(get_db)
):
    """
    Get the currently active trip for a vehicle, if any.
    """
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == True)
    )
    trip = result.scalar_one_or_none()
    return trip


@router.get("/details/{trip_id}", response_model=TripResponse)
async def get_trip_details(
    trip_id: UUID,
    db: AsyncSession=Depends(get_db)
):
    """
    Get detailed information about a specific trip.
    """
    result = await db.execute(
        select(Trip).where(Trip.id == trip_id)
    )
    trip = result.scalar_one_or_none()
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    return trip
