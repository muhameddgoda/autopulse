"""
AutoPulse Telemetry API Routes
REST endpoints + WebSocket for real-time streaming
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID
from typing import List
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.database import get_db
from app.models.telemetry import Vehicle, TelemetryReading, Trip
from app.schemas.telemetry import (
    VehicleResponse,
    TelemetryReadingCreate,
    TelemetryReadingResponse,
    TripCreate,
    TripEnd,
    TripResponse,
)

router = APIRouter(prefix="/api/telemetry", tags=["Telemetry"])


# Helper function to get current UTC time with timezone
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================
# CONNECTION MANAGER FOR WEBSOCKETS
# ============================================

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming"""
    
    def __init__(self):
        self.active_connections: dict[UUID, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, vehicle_id: UUID):
        await websocket.accept()
        if vehicle_id not in self.active_connections:
            self.active_connections[vehicle_id] = []
        self.active_connections[vehicle_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, vehicle_id: UUID):
        if vehicle_id in self.active_connections:
            self.active_connections[vehicle_id].remove(websocket)
            if not self.active_connections[vehicle_id]:
                del self.active_connections[vehicle_id]
    
    async def broadcast_to_vehicle(self, vehicle_id: UUID, message: dict):
        """Send telemetry update to all clients watching this vehicle"""
        if vehicle_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[vehicle_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for conn in dead_connections:
                self.disconnect(conn, vehicle_id)


manager = ConnectionManager()


# ============================================
# VEHICLE ENDPOINTS
# ============================================

@router.get("/vehicles", response_model=List[VehicleResponse])
async def get_vehicles(db: AsyncSession = Depends(get_db)):
    """Get all registered vehicles"""
    result = await db.execute(select(Vehicle))
    vehicles = result.scalars().all()
    return vehicles


@router.get("/vehicles/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get a specific vehicle by ID"""
    result = await db.execute(select(Vehicle).where(Vehicle.id == vehicle_id))
    vehicle = result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle


# ============================================
# TELEMETRY ENDPOINTS
# ============================================

@router.post("/reading", response_model=TelemetryReadingResponse)
async def create_reading(reading: TelemetryReadingCreate, db: AsyncSession = Depends(get_db)):
    """
    Store a new telemetry reading.
    This endpoint is called by the simulator.
    Also updates active trip statistics in real-time.
    """
    # Verify vehicle exists
    result = await db.execute(select(Vehicle).where(Vehicle.id == reading.vehicle_id))
    vehicle = result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Get reading data and exclude any fields not in the SQLAlchemy model
    reading_data = reading.model_dump()
    # Remove 'timestamp' if present - the DB column is 'time', not 'timestamp'
    reading_data.pop('timestamp', None)
    
    # Create reading
    db_reading = TelemetryReading(
        time=utc_now(),
        **reading_data
    )
    db.add(db_reading)
    
    # Update active trip if exists
    trip_result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == reading.vehicle_id)
        .where(Trip.is_active == True)
    )
    active_trip = trip_result.scalar_one_or_none()
    
    if active_trip:
        # Update running totals
        active_trip.total_readings = (active_trip.total_readings or 0) + 1
        active_trip.speed_sum = (active_trip.speed_sum or 0) + (reading.speed_kmh or 0)
        active_trip.rpm_sum = (active_trip.rpm_sum or 0) + (reading.rpm or 0)
        
        # Update max values
        if reading.speed_kmh and (not active_trip.max_speed_kmh or reading.speed_kmh > active_trip.max_speed_kmh):
            active_trip.max_speed_kmh = reading.speed_kmh
        if reading.rpm and (not active_trip.max_rpm or reading.rpm > active_trip.max_rpm):
            active_trip.max_rpm = reading.rpm
        
        # Update mode breakdown (assuming 1 second between readings)
        mode = reading.driving_mode
        if mode:
            if mode == 'parked':
                active_trip.mode_parked_seconds = (active_trip.mode_parked_seconds or 0) + 1
            elif mode == 'city':
                active_trip.mode_city_seconds = (active_trip.mode_city_seconds or 0) + 1
            elif mode == 'highway':
                active_trip.mode_highway_seconds = (active_trip.mode_highway_seconds or 0) + 1
            elif mode == 'sport':
                active_trip.mode_sport_seconds = (active_trip.mode_sport_seconds or 0) + 1
            elif mode == 'reverse':
                active_trip.mode_reverse_seconds = (active_trip.mode_reverse_seconds or 0) + 1
        
        # Store fuel level (for fuel consumption calc at trip end)
        if reading.fuel_level:
            if active_trip.fuel_start is None:
                active_trip.fuel_start = reading.fuel_level
            active_trip.fuel_end = reading.fuel_level
    
    await db.commit()
    await db.refresh(db_reading)
    
    # Broadcast to WebSocket clients
    await manager.broadcast_to_vehicle(
        reading.vehicle_id,
        {
            "type": "telemetry_update",
            "data": {
                "timestamp": db_reading.time.isoformat(),
                **reading.model_dump(mode="json")
            }
        }
    )
    
    return db_reading


@router.get("/latest/{vehicle_id}", response_model=TelemetryReadingResponse | None)
async def get_latest_reading(vehicle_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get the most recent telemetry reading for a vehicle"""
    result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
        .order_by(desc(TelemetryReading.time))
        .limit(1)
    )
    reading = result.scalar_one_or_none()
    return reading


@router.get("/history/{vehicle_id}", response_model=List[TelemetryReadingResponse])
async def get_reading_history(
    vehicle_id: UUID,
    minutes: int = Query(default=5, ge=1, le=60),
    db: AsyncSession = Depends(get_db)
):
    """Get telemetry history for the last N minutes"""
    since = utc_now() - timedelta(minutes=minutes)
    
    result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
        .where(TelemetryReading.time >= since)
        .order_by(TelemetryReading.time)
    )
    readings = result.scalars().all()
    return readings


# ============================================
# TRIP ENDPOINTS
# ============================================

@router.post("/trips/start", response_model=TripResponse)
async def start_trip(trip: TripCreate, db: AsyncSession = Depends(get_db)):
    """Start a new trip"""
    # Check if there's already an active trip
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == trip.vehicle_id)
        .where(Trip.is_active == True)
    )
    active_trip = result.scalar_one_or_none()
    if active_trip:
        raise HTTPException(status_code=400, detail="Vehicle already has an active trip")
    
    # Create new trip
    db_trip = Trip(
        vehicle_id=trip.vehicle_id,
        start_time=utc_now(),
        start_latitude=trip.start_latitude,
        start_longitude=trip.start_longitude,
    )
    db.add(db_trip)
    await db.commit()
    await db.refresh(db_trip)
    
    # Broadcast trip start
    await manager.broadcast_to_vehicle(
        trip.vehicle_id,
        {
            "type": "trip_started",
            "data": {"trip_id": str(db_trip.id), "start_time": db_trip.start_time.isoformat()}
        }
    )
    
    return db_trip


@router.post("/trips/{trip_id}/end", response_model=TripResponse)
async def end_trip(trip_id: UUID, trip_end: TripEnd, db: AsyncSession = Depends(get_db)):
    """End an active trip and calculate final statistics"""
    result = await db.execute(select(Trip).where(Trip.id == trip_id))
    trip = result.scalar_one_or_none()
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    if not trip.is_active:
        raise HTTPException(status_code=400, detail="Trip is already ended")
    
    # Finalize trip
    end_time = utc_now()
    trip.end_time = end_time
    trip.is_active = False
    trip.end_latitude = trip_end.end_latitude
    trip.end_longitude = trip_end.end_longitude
    
    # FIXED: Use number of readings as duration (each reading = ~1 second of simulated driving)
    # This gives accurate results for both real-time and batch-generated training data
    wall_clock_duration = int((end_time - trip.start_time).total_seconds())
    readings_based_duration = trip.total_readings or 0
    
    # Use the larger of the two - handles both real-time driving and training data generation
    trip.duration_seconds = max(wall_clock_duration, readings_based_duration)
    
    # Calculate averages from running totals
    if trip.total_readings and trip.total_readings > 0:
        trip.avg_speed_kmh = trip.speed_sum / trip.total_readings if trip.speed_sum else 0
        trip.avg_rpm = int(trip.rpm_sum / trip.total_readings) if trip.rpm_sum else 0
    
    # Calculate distance (avg_speed * duration)
    if trip.avg_speed_kmh and trip.duration_seconds:
        trip.distance_km = (trip.avg_speed_kmh * trip.duration_seconds) / 3600
    
    # Calculate fuel used
    if trip.fuel_start is not None and trip.fuel_end is not None:
        # Assuming 64L tank for Porsche 911
        tank_capacity = 64.0
        fuel_used_percent = trip.fuel_start - trip.fuel_end
        trip.fuel_used_liters = (fuel_used_percent / 100) * tank_capacity
    
    await db.commit()
    await db.refresh(trip)
    
    # Broadcast trip end with full stats
    await manager.broadcast_to_vehicle(
        trip.vehicle_id,
        {
            "type": "trip_ended",
            "data": {
                "trip_id": str(trip.id),
                "distance_km": trip.distance_km,
                "duration_seconds": trip.duration_seconds,
                "avg_speed_kmh": trip.avg_speed_kmh,
                "max_speed_kmh": trip.max_speed_kmh,
                "avg_rpm": trip.avg_rpm,
                "max_rpm": trip.max_rpm,
                "fuel_used_liters": trip.fuel_used_liters,
                "mode_breakdown": {
                    "parked": trip.mode_parked_seconds,
                    "city": trip.mode_city_seconds,
                    "highway": trip.mode_highway_seconds,
                    "sport": trip.mode_sport_seconds,
                    "reverse": trip.mode_reverse_seconds,
                }
            }
        }
    )
    
    return trip


@router.get("/trips/active/{vehicle_id}", response_model=TripResponse | None)
async def get_active_trip(vehicle_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get the current active trip for a vehicle"""
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == True)
    )
    return result.scalar_one_or_none()


@router.post("/trips/end-active/{vehicle_id}", response_model=TripResponse | None)
async def end_active_trip(vehicle_id: UUID, db: AsyncSession = Depends(get_db)):
    """End any active trip for a vehicle (convenience endpoint)"""
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == True)
    )
    trip = result.scalar_one_or_none()
    
    if not trip:
        return None
    
    # Finalize trip
    end_time = utc_now()
    trip.end_time = end_time
    trip.is_active = False
    
    # FIXED: Use number of readings as duration
    wall_clock_duration = int((end_time - trip.start_time).total_seconds())
    readings_based_duration = trip.total_readings or 0
    trip.duration_seconds = max(wall_clock_duration, readings_based_duration)
    
    # Calculate averages from running totals
    if trip.total_readings and trip.total_readings > 0:
        trip.avg_speed_kmh = trip.speed_sum / trip.total_readings if trip.speed_sum else 0
        trip.avg_rpm = int(trip.rpm_sum / trip.total_readings) if trip.rpm_sum else 0
    
    # Calculate distance (avg_speed * duration)
    if trip.avg_speed_kmh and trip.duration_seconds:
        trip.distance_km = (trip.avg_speed_kmh * trip.duration_seconds) / 3600
    
    # Calculate fuel used
    if trip.fuel_start is not None and trip.fuel_end is not None:
        tank_capacity = 64.0
        fuel_used_percent = trip.fuel_start - trip.fuel_end
        trip.fuel_used_liters = (fuel_used_percent / 100) * tank_capacity
    
    await db.commit()
    await db.refresh(trip)
    
    return trip


@router.get("/trips/{vehicle_id}", response_model=List[TripResponse])
async def get_trips(
    vehicle_id: UUID,
    limit: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get trip history for a vehicle"""
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .order_by(desc(Trip.start_time))
        .limit(limit)
    )
    return result.scalars().all()


# ============================================
# ANALYTICS & EXPORT ENDPOINTS
# ============================================

@router.get("/stats/weekly/{vehicle_id}")
async def get_weekly_stats(vehicle_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get weekly driving statistics"""
    # Get trips from the last 7 days
    week_ago = utc_now() - timedelta(days=7)
    
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.start_time >= week_ago)
        .where(Trip.is_active == False)
        .order_by(Trip.start_time)
    )
    trips = result.scalars().all()
    
    # Calculate weekly totals
    total_distance = sum(t.distance_km or 0 for t in trips)
    total_duration = sum(t.duration_seconds or 0 for t in trips)
    total_fuel = sum(t.fuel_used_liters or 0 for t in trips)
    
    # Mode breakdown totals
    mode_totals = {
        "parked": sum(t.mode_parked_seconds or 0 for t in trips),
        "city": sum(t.mode_city_seconds or 0 for t in trips),
        "highway": sum(t.mode_highway_seconds or 0 for t in trips),
        "sport": sum(t.mode_sport_seconds or 0 for t in trips),
        "reverse": sum(t.mode_reverse_seconds or 0 for t in trips),
    }
    
    # Daily breakdown
    daily_stats = {}
    for trip in trips:
        day = trip.start_time.strftime("%Y-%m-%d")
        if day not in daily_stats:
            daily_stats[day] = {"distance_km": 0, "trips": 0, "duration_seconds": 0}
        daily_stats[day]["distance_km"] += trip.distance_km or 0
        daily_stats[day]["trips"] += 1
        daily_stats[day]["duration_seconds"] += trip.duration_seconds or 0
    
    # Calculate averages
    avg_speed = 0
    if trips:
        speeds = [t.avg_speed_kmh for t in trips if t.avg_speed_kmh]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
    
    return {
        "period": "7_days",
        "total_trips": len(trips),
        "total_distance_km": round(total_distance, 2),
        "total_duration_seconds": total_duration,
        "total_fuel_liters": round(total_fuel, 2),
        "avg_speed_kmh": round(avg_speed, 1),
        "mode_breakdown_seconds": mode_totals,
        "daily_breakdown": daily_stats,
    }


@router.get("/export/csv/{vehicle_id}")
async def export_telemetry_csv(
    vehicle_id: UUID,
    hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_db)
):
    """Export telemetry data as CSV for ML training"""
    from fastapi.responses import StreamingResponse
    import io
    import csv
    
    since = utc_now() - timedelta(hours=hours)
    
    result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
        .where(TelemetryReading.time >= since)
        .order_by(TelemetryReading.time)
    )
    readings = result.scalars().all()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header with ML metrics
    writer.writerow([
        "timestamp",
        "speed_kmh",
        "rpm",
        "gear",
        "throttle_position",
        "engine_temp",
        "oil_temp",
        "oil_pressure",
        "fuel_level",
        "battery_voltage",
        "tire_pressure_fl",
        "tire_pressure_fr",
        "tire_pressure_rl",
        "tire_pressure_rr",
        "latitude",
        "longitude",
        "heading",
        "driving_mode",
        # ML metrics
        "acceleration_ms2",
        "acceleration_g",
        "jerk_ms3",
        "is_harsh_braking",
        "is_harsh_acceleration",
        "is_over_rpm",
        "is_speeding",
        "is_idling",
        "engine_stress_score",
    ])
    
    # Data rows
    for r in readings:
        writer.writerow([
            r.time.isoformat(),
            r.speed_kmh,
            r.rpm,
            r.gear,
            r.throttle_position,
            r.engine_temp,
            r.oil_temp,
            r.oil_pressure,
            r.fuel_level,
            r.battery_voltage,
            r.tire_pressure_fl,
            r.tire_pressure_fr,
            r.tire_pressure_rl,
            r.tire_pressure_rr,
            r.latitude,
            r.longitude,
            r.heading,
            r.driving_mode,
            # ML metrics (may be None for old data)
            getattr(r, 'acceleration_ms2', None),
            getattr(r, 'acceleration_g', None),
            getattr(r, 'jerk_ms3', None),
            getattr(r, 'is_harsh_braking', False),
            getattr(r, 'is_harsh_acceleration', False),
            getattr(r, 'is_over_rpm', False),
            getattr(r, 'is_speeding', False),
            getattr(r, 'is_idling', False),
            getattr(r, 'engine_stress_score', None),
        ])
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=telemetry_{vehicle_id}_{hours}h.csv"
        }
    )


@router.get("/export/trips-csv/{vehicle_id}")
async def export_trips_csv(
    vehicle_id: UUID,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Export trip data as CSV for maintenance prediction ML"""
    from fastapi.responses import StreamingResponse
    import io
    import csv
    
    since = utc_now() - timedelta(days=days)
    
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.start_time >= since)
        .where(Trip.is_active == False)
        .order_by(Trip.start_time)
    )
    trips = result.scalars().all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "trip_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "distance_km",
        "avg_speed_kmh",
        "max_speed_kmh",
        "avg_rpm",
        "max_rpm",
        "fuel_used_liters",
        "mode_parked_seconds",
        "mode_city_seconds",
        "mode_highway_seconds",
        "mode_sport_seconds",
        "mode_reverse_seconds",
    ])
    
    for t in trips:
        writer.writerow([
            str(t.id),
            t.start_time.isoformat() if t.start_time else "",
            t.end_time.isoformat() if t.end_time else "",
            t.duration_seconds,
            t.distance_km,
            t.avg_speed_kmh,
            t.max_speed_kmh,
            t.avg_rpm,
            t.max_rpm,
            t.fuel_used_liters,
            t.mode_parked_seconds,
            t.mode_city_seconds,
            t.mode_highway_seconds,
            t.mode_sport_seconds,
            t.mode_reverse_seconds,
        ])
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=trips_{vehicle_id}_{days}d.csv"
        }
    )


# ============================================
# WEBSOCKET ENDPOINT
# ============================================

@router.websocket("/stream/{vehicle_id}")
async def websocket_telemetry_stream(websocket: WebSocket, vehicle_id: UUID):
    """
    WebSocket endpoint for real-time telemetry streaming.
    Connect to receive live updates for a specific vehicle.
    """
    await manager.connect(websocket, vehicle_id)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "vehicle_id": str(vehicle_id),
            "message": "Listening for telemetry updates"
        })
        
        # Keep connection alive, handle any client messages
        while True:
            try:
                # Wait for any client message (ping/pong, commands, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, vehicle_id)
    except Exception as e:
        manager.disconnect(websocket, vehicle_id)
        raise


# ============================================
# ML / DRIVER SCORING ENDPOINTS
# ============================================

# Import ML service
try:
    from app.ml.api.routes import ml_service
    ML_AVAILABLE = True
    print("✅ ML service loaded successfully")
except ImportError as e:
    print(f"❌ ML import error: {e}")
    ML_AVAILABLE = False


@router.get("/ml/score/trip/{trip_id}")
async def get_trip_score(
    trip_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get driver score for a completed trip
    
    Analyzes all telemetry readings from the trip and returns:
    - Overall score (0-100)
    - Behavior classification (calm/normal/aggressive)
    - Component scores breakdown
    - Insights and recommendations
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    # Get trip info
    trip_result = await db.execute(
        select(Trip).where(Trip.id == trip_id)
    )
    trip = trip_result.scalar_one_or_none()
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    if trip.is_active:
        raise HTTPException(status_code=400, detail="Trip is still active. End the trip first.")
    
    # Get telemetry readings for the trip
    result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == trip.vehicle_id)
        .where(TelemetryReading.time >= trip.start_time)
        .where(TelemetryReading.time <= trip.end_time)
        .order_by(TelemetryReading.time)
    )
    readings_orm = result.scalars().all()
    
    # Fallback: if time-based query didn't get enough readings, use total_readings limit
    if len(readings_orm) < 30 and trip.total_readings and trip.total_readings >= 30:
        result = await db.execute(
            select(TelemetryReading)
            .where(TelemetryReading.vehicle_id == trip.vehicle_id)
            .where(TelemetryReading.time >= trip.start_time)
            .order_by(TelemetryReading.time)
            .limit(trip.total_readings)
        )
        readings_orm = result.scalars().all()
    
    if not readings_orm:
        raise HTTPException(status_code=404, detail="No telemetry data found for this trip")
    
    # Convert to dictionaries
    readings = []
    for r in readings_orm:
        readings.append({
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
        })
    
    # Analyze trip
    analysis = ml_service.analyze_trip(
        readings,
        trip_id=str(trip_id),
        vehicle_id=str(trip.vehicle_id)
    )
    
    return analysis


@router.get("/ml/score/vehicle/{vehicle_id}")
async def get_vehicle_scores(
    vehicle_id: UUID,
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """
    Get driver scores for recent trips of a vehicle
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    # Get recent completed trips
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == False)
        .order_by(desc(Trip.end_time))
        .limit(limit)
    )
    trips = result.scalars().all()
    
    scores = []
    for trip in trips:
        # Get readings for each trip
        readings_result = await db.execute(
            select(TelemetryReading)
            .where(TelemetryReading.vehicle_id == vehicle_id)
            .where(TelemetryReading.time >= trip.start_time)
            .where(TelemetryReading.time <= trip.end_time)
            .order_by(TelemetryReading.time)
        )
        readings_orm = readings_result.scalars().all()
        
        # Fallback: if time-based query didn't get enough readings, use total_readings limit
        if len(readings_orm) < 30 and trip.total_readings and trip.total_readings >= 30:
            readings_result = await db.execute(
                select(TelemetryReading)
                .where(TelemetryReading.vehicle_id == vehicle_id)
                .where(TelemetryReading.time >= trip.start_time)
                .order_by(TelemetryReading.time)
                .limit(trip.total_readings)
            )
            readings_orm = readings_result.scalars().all()
        
        if len(readings_orm) < 30:
            continue
        
        readings = [{
            "timestamp": r.time.isoformat() if r.time else None,
            "speed_kmh": r.speed_kmh,
            "rpm": r.rpm,
            "gear": r.gear,
            "throttle_position": r.throttle_position,
            "engine_temp": r.engine_temp,
            "acceleration_g": r.acceleration_g or 0,
            "is_harsh_braking": r.is_harsh_braking or False,
            "is_harsh_acceleration": r.is_harsh_acceleration or False,
            "is_speeding": r.is_speeding or False,
            "is_idling": r.is_idling or False,
            "driving_mode": r.driving_mode,
            "engine_stress_score": r.engine_stress_score or 0,
        } for r in readings_orm]
        
        analysis = ml_service.analyze_trip(
            readings,
            trip_id=str(trip.id),
            vehicle_id=str(vehicle_id)
        )
        
        if analysis and "error" not in analysis:
            analysis["trip_start"] = trip.start_time.isoformat() if trip.start_time else None
            analysis["trip_end"] = trip.end_time.isoformat() if trip.end_time else None
            scores.append(analysis)
    
    return {
        "vehicle_id": str(vehicle_id),
        "trips_analyzed": len(scores),
        "scores": scores
    }


@router.get("/ml/summary/{vehicle_id}")
async def get_driver_summary(
    vehicle_id: UUID,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """
    Get aggregated driver summary and statistics
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    since = utc_now() - timedelta(days=days)
    
    # Get trips in date range
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == False)
        .where(Trip.start_time >= since)
        .order_by(desc(Trip.end_time))
    )
    trips = result.scalars().all()
    
    # Analyze each trip
    trip_scores = []
    for trip in trips:
        readings_result = await db.execute(
            select(TelemetryReading)
            .where(TelemetryReading.vehicle_id == vehicle_id)
            .where(TelemetryReading.time >= trip.start_time)
            .where(TelemetryReading.time <= trip.end_time)
            .order_by(TelemetryReading.time)
        )
        readings_orm = readings_result.scalars().all()
        
        # Fallback: if time-based query didn't get enough readings, use total_readings limit
        if len(readings_orm) < 30 and trip.total_readings and trip.total_readings >= 30:
            readings_result = await db.execute(
                select(TelemetryReading)
                .where(TelemetryReading.vehicle_id == vehicle_id)
                .where(TelemetryReading.time >= trip.start_time)
                .order_by(TelemetryReading.time)
                .limit(trip.total_readings)
            )
            readings_orm = readings_result.scalars().all()
        
        if len(readings_orm) < 30:
            continue
        
        readings = [{
            "timestamp": r.time.isoformat() if r.time else None,
            "speed_kmh": r.speed_kmh,
            "rpm": r.rpm,
            "throttle_position": r.throttle_position,
            "acceleration_g": r.acceleration_g or 0,
            "is_harsh_braking": r.is_harsh_braking or False,
            "is_harsh_acceleration": r.is_harsh_acceleration or False,
            "is_speeding": r.is_speeding or False,
            "is_idling": r.is_idling or False,
            "driving_mode": r.driving_mode,
            "engine_stress_score": r.engine_stress_score or 0,
        } for r in readings_orm]
        
        analysis = ml_service.analyze_trip(readings, str(trip.id), str(vehicle_id))
        if analysis and "error" not in analysis:
            trip_scores.append(analysis)
    
    # Generate summary
    summary = ml_service.get_driver_summary(trip_scores)
    summary["vehicle_id"] = str(vehicle_id)
    summary["period_days"] = days
    
    return summary


@router.get("/debug/training-data/{vehicle_id}")
async def debug_training_data(vehicle_id: UUID, db: AsyncSession = Depends(get_db)):
    """
    Debug endpoint to check training data state
    """
    from datetime import timedelta
    
    since = utc_now() - timedelta(days=30)
    
    # Get trips
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == False)
        .where(Trip.start_time >= since)
        .order_by(Trip.start_time)
    )
    trips = result.scalars().all()
    
    # Get all readings count
    readings_result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
    )
    all_readings = readings_result.scalars().all()
    
    # Analyze trips
    trip_info = []
    total_expected_readings = 0
    for t in trips[:10]:  # First 10 trips
        total_expected_readings += (t.total_readings or 0)
        trip_info.append({
            "id": str(t.id)[:8],
            "total_readings": t.total_readings,
            "distance_km": t.distance_km,
            "avg_speed_kmh": t.avg_speed_kmh,
            "duration_seconds": t.duration_seconds,
        })
    
    return {
        "total_trips": len(trips),
        "total_readings_in_db": len(all_readings),
        "sum_of_trip_total_readings": sum(t.total_readings or 0 for t in trips),
        "first_10_trips": trip_info,
        "alignment_check": {
            "readings_in_db": len(all_readings),
            "expected_from_trips": sum(t.total_readings or 0 for t in trips),
            "aligned": len(all_readings) == sum(t.total_readings or 0 for t in trips)
        }
    }


# ============================================
# ML TRAINING ENDPOINTS
# ============================================

@router.post("/ml/train/{vehicle_id}")
async def train_ml_models(
    vehicle_id: UUID,
    days: int = Query(default=30, ge=1, le=365),
    train_behavior: bool = Query(default=True),
    train_anomaly: bool = Query(default=True),
    db: AsyncSession = Depends(get_db)
):
    """
    Train ML models using historical trip data
    
    This endpoint:
    1. Collects all completed trips for the vehicle
    2. Extracts features from each trip
    3. Uses rule-based scores as labels
    4. Trains behavior classifier and/or anomaly detector
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        from app.ml.training import DataCollector, data_collector, behavior_classifier, anomaly_detector
        from app.ml.features.extractor import FeatureExtractor
        from app.ml.models.driver_scorer import DriverScorer
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"ML training modules not available: {e}")
    
    since = utc_now() - timedelta(days=days)
    
    # Get completed trips ordered by start_time
    result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == False)
        .where(Trip.start_time >= since)
        .order_by(Trip.start_time)
    )
    trips = result.scalars().all()
    
    if len(trips) < 10:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least 10 trips for training, found {len(trips)}"
        )
    
    # Get ALL readings for the vehicle, ordered by time
    # We'll partition them based on trip.total_readings
    all_readings_result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
        .order_by(TelemetryReading.time)
    )
    all_readings_orm = all_readings_result.scalars().all()
    
    if not all_readings_orm:
        raise HTTPException(status_code=400, detail="No telemetry readings found")
    
    # Convert all readings to dicts once
    all_readings = [{
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
    } for r in all_readings_orm]
    
    # Partition readings by trip
    # Since trips were created in order and readings were inserted in order,
    # we can use total_readings to slice the readings array
    feature_extractor = FeatureExtractor()
    scorer = DriverScorer()
    examples = []
    
    reading_offset = 0
    skipped_trips = 0
    feature_extraction_failed = 0
    
    for trip in trips:
        trip_reading_count = trip.total_readings or 0
        
        if trip_reading_count < 30:
            # Skip trips with too few readings
            reading_offset += trip_reading_count
            skipped_trips += 1
            continue
        
        # Get this trip's readings from the ordered list
        trip_readings = all_readings[reading_offset:reading_offset + trip_reading_count]
        reading_offset += trip_reading_count
        
        if len(trip_readings) < 30:
            skipped_trips += 1
            continue
        
        # IMPORTANT: For batch-generated training data, timestamps are wall-clock time
        # which doesn't reflect the actual simulated driving duration.
        # Override the first/last timestamps to create a proper duration span.
        # Each reading = 1 second of simulated driving.
        if trip_readings:
            # Create synthetic timestamps that span the correct duration
            from datetime import timedelta as td
            base_time = datetime.fromisoformat(trip_readings[0]["timestamp"].replace('Z', '+00:00')) if trip_readings[0].get("timestamp") else utc_now()
            for i, reading in enumerate(trip_readings):
                # Each reading represents 1 second
                reading["timestamp"] = (base_time + td(seconds=i)).isoformat()
        
        # Extract features
        features = feature_extractor.extract_from_readings(
            trip_readings, str(trip.id), str(vehicle_id)
        )
        
        if features is None:
            feature_extraction_failed += 1
            continue
        
        # Get rule-based score for labeling
        rule_score = scorer.score_trip(features)
        
        # Create training example
        example = data_collector.features_to_training_example(
            features,
            behavior_label=rule_score.behavior_label.value,
            risk_label=rule_score.risk_level.value,
            rule_score=rule_score.total_score
        )
        examples.append(example)
    
    if len(examples) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(examples)} valid examples. Skipped {skipped_trips} trips (too few readings), {feature_extraction_failed} feature extraction failures. Total trips: {len(trips)}"
        )
    
    # Save training data
    data_collector.save_training_data(examples, f"training_data_{vehicle_id}.csv")
    
    # Train models
    results = {
        "vehicle_id": str(vehicle_id),
        "training_examples": len(examples),
        "trips_processed": len(trips),
    }
    
    if train_behavior:
        behavior_result = behavior_classifier.train(examples)
        if behavior_result.get("success"):
            behavior_classifier.save(f"behavior_{vehicle_id}")
        results["behavior_classifier"] = behavior_result
    
    if train_anomaly:
        anomaly_result = anomaly_detector.train(examples)
        if anomaly_result.get("success"):
            anomaly_detector.save(f"anomaly_{vehicle_id}")
        results["anomaly_detector"] = anomaly_result
    
    return results


@router.get("/ml/models/status")
async def get_ml_models_status():
    """
    Get status of trained ML models
    """
    if not ML_AVAILABLE:
        return {"ml_available": False, "error": "ML service not available"}
    
    try:
        from app.ml.training import behavior_classifier, anomaly_detector
        from pathlib import Path
        
        model_dir = Path("app/ml/trained_models")
        
        status = {
            "ml_available": True,
            "sklearn_available": True,
            "model_directory": str(model_dir),
            "models": {}
        }
        
        # Check for saved models
        if model_dir.exists():
            for model_file in model_dir.glob("*.joblib"):
                name = model_file.stem
                metadata_file = model_dir / f"{name.replace('.joblib', '')}_metadata.json"
                
                model_info = {
                    "file": str(model_file),
                    "exists": True,
                }
                
                if metadata_file.exists():
                    import json
                    with open(metadata_file) as f:
                        model_info["metadata"] = json.load(f)
                
                status["models"][name] = model_info
        
        return status
        
    except ImportError as e:
        return {"ml_available": True, "sklearn_available": False, "error": str(e)}


@router.post("/ml/score/hybrid/{trip_id}")
async def get_hybrid_score(
    trip_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get hybrid score (rules + ML) for a trip
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        from app.ml.models.hybrid_scorer import hybrid_scorer
        from app.ml.features.extractor import FeatureExtractor
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Hybrid scorer not available: {e}")
    
    # Get trip
    trip_result = await db.execute(
        select(Trip).where(Trip.id == trip_id)
    )
    trip = trip_result.scalar_one_or_none()
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    if trip.is_active:
        raise HTTPException(status_code=400, detail="Trip is still active")
    
    # Get telemetry
    result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == trip.vehicle_id)
        .where(TelemetryReading.time >= trip.start_time)
        .where(TelemetryReading.time <= trip.end_time)
        .order_by(TelemetryReading.time)
    )
    readings_orm = result.scalars().all()
    
    # Fallback: if time-based query didn't get enough readings, use total_readings limit
    if len(readings_orm) < 30 and trip.total_readings and trip.total_readings >= 30:
        result = await db.execute(
            select(TelemetryReading)
            .where(TelemetryReading.vehicle_id == trip.vehicle_id)
            .where(TelemetryReading.time >= trip.start_time)
            .order_by(TelemetryReading.time)
            .limit(trip.total_readings)
        )
        readings_orm = result.scalars().all()
    
    if not readings_orm:
        raise HTTPException(status_code=404, detail="No telemetry data")
    
    # Convert
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
    features = feature_extractor.extract_from_readings(
        readings, str(trip_id), str(trip.vehicle_id)
    )
    
    if features is None:
        raise HTTPException(status_code=400, detail="Insufficient data")
    
    # Load ML models for this vehicle
    hybrid_scorer.load_models_for_vehicle(str(trip.vehicle_id))
    
    # Get hybrid score
    hybrid_result = hybrid_scorer.score_trip(features)
    
    return hybrid_scorer.score_to_dict(hybrid_result)