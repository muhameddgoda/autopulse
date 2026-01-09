"""
AutoPulse Telemetry API - SLIM VERSION
======================================
This module handles ONLY:
- Telemetry data ingestion (POST readings)
- Telemetry history retrieval
- WebSocket connections for real-time data
- ML model status and training endpoints

Other functionality has been moved to:
- api/vehicles.py    - Vehicle CRUD
- api/trips.py       - Trip lifecycle
- api/analytics.py   - Stats, exports
- api/scoring.py     - All scoring logic
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from uuid import UUID
import asyncio

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func

from app.database import get_db
from app.models.telemetry import Vehicle, TelemetryReading, Trip
from app.schemas.telemetry import (
    TelemetryReadingCreate,
    TelemetryReadingResponse,
    TelemetryBatchCreate
)

router = APIRouter()


def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)

# ============================================
# WebSocket Connection Manager
# ============================================


class ConnectionManager:
    """Manages WebSocket connections for real-time telemetry"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, vehicle_id: str):
        await websocket.accept()
        if vehicle_id not in self.active_connections:
            self.active_connections[vehicle_id] = []
        self.active_connections[vehicle_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, vehicle_id: str):
        if vehicle_id in self.active_connections:
            if websocket in self.active_connections[vehicle_id]:
                self.active_connections[vehicle_id].remove(websocket)
            if not self.active_connections[vehicle_id]:
                del self.active_connections[vehicle_id]
    
    async def broadcast_to_vehicle(self, vehicle_id: str, data: dict):
        if vehicle_id in self.active_connections:
            for connection in self.active_connections[vehicle_id]:
                try:
                    await connection.send_json(data)
                except Exception:
                    pass  # Connection may be closed


manager = ConnectionManager()

# ============================================
# TELEMETRY INGESTION
# ============================================


@router.post("/readings", response_model=TelemetryReadingResponse)
@router.post("/reading", response_model=TelemetryReadingResponse, include_in_schema=False)  # Compat
async def create_telemetry_reading(
    reading: TelemetryReadingCreate,
    db: AsyncSession=Depends(get_db)
):
    """
    Ingest a single telemetry reading and update active trip stats
    """
    # Verify vehicle exists
    vehicle_result = await db.execute(
        select(Vehicle).where(Vehicle.id == reading.vehicle_id)
    )
    vehicle = vehicle_result.scalar_one_or_none()
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Create reading
    db_reading = TelemetryReading(
        vehicle_id=reading.vehicle_id,
        time=reading.time or utc_now(),
        speed_kmh=reading.speed_kmh,
        rpm=reading.rpm,
        gear=reading.gear,
        throttle_position=reading.throttle_position,
        engine_temp=reading.engine_temp,
        oil_temp=reading.oil_temp,
        oil_pressure=reading.oil_pressure,
        fuel_level=reading.fuel_level,
        acceleration_g=reading.acceleration_g,
        is_harsh_braking=reading.is_harsh_braking,
        is_harsh_acceleration=reading.is_harsh_acceleration,
        is_speeding=reading.is_speeding,
        is_idling=reading.is_idling,
        driving_mode=reading.driving_mode,
        engine_stress_score=reading.engine_stress_score
    )
    db.add(db_reading)
    
    # Update active trip if exists
    active_trip_result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == reading.vehicle_id)
        .where(Trip.is_active == True)
    )
    active_trip = active_trip_result.scalar_one_or_none()
    
    if active_trip:
        # Increment reading count
        active_trip.total_readings = (active_trip.total_readings or 0) + 1
        
        # Update speed sum for average calculation
        active_trip.speed_sum = (active_trip.speed_sum or 0) + reading.speed_kmh
        
        # Update RPM sum for average calculation
        active_trip.rpm_sum = (active_trip.rpm_sum or 0) + (reading.rpm or 0)
        
        # Update trip stats
        if active_trip.max_speed_kmh is None or reading.speed_kmh > active_trip.max_speed_kmh:
            active_trip.max_speed_kmh = reading.speed_kmh
        
        # Track max RPM
        if reading.rpm:
            if active_trip.max_rpm is None or reading.rpm > active_trip.max_rpm:
                active_trip.max_rpm = reading.rpm
        
        # Calculate distance: speed (km/h) * time (1 second = 1/3600 hour)
        # Assuming readings come every ~1 second
        distance_increment = reading.speed_kmh / 3600.0
        active_trip.distance_km = (active_trip.distance_km or 0) + distance_increment
        
        # Track driving mode time (assuming 1 second per reading)
        if reading.driving_mode:
            mode = reading.driving_mode.lower()
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
        
        # Set start fuel if not set
        if active_trip.fuel_start is None and reading.fuel_level:
            active_trip.fuel_start = reading.fuel_level
        
        # Update end fuel and calculate consumed
        if reading.fuel_level:
            active_trip.fuel_end = reading.fuel_level
            if active_trip.fuel_start:
                active_trip.fuel_used_liters = active_trip.fuel_start - reading.fuel_level
        
        # Track harsh events
        if reading.is_harsh_braking:
            active_trip.harsh_brake_count = (active_trip.harsh_brake_count or 0) + 1
        if reading.is_harsh_acceleration:
            active_trip.harsh_accel_count = (active_trip.harsh_accel_count or 0) + 1
    
    await db.commit()
    await db.refresh(db_reading)
    
    # Broadcast to WebSocket clients
    await manager.broadcast_to_vehicle(
        str(reading.vehicle_id),
        {
            "type": "telemetry_update",
            "data": {
                "vehicle_id": str(db_reading.vehicle_id),
                "time": db_reading.time.isoformat() if db_reading.time else None,
                "speed_kmh": db_reading.speed_kmh,
                "rpm": db_reading.rpm,
                "gear": db_reading.gear,
                "throttle_position": db_reading.throttle_position,
                "engine_temp": db_reading.engine_temp,
                "oil_temp": db_reading.oil_temp,
                "oil_pressure": db_reading.oil_pressure,
                "fuel_level": db_reading.fuel_level,
                "battery_voltage": db_reading.battery_voltage,
                "tire_pressure_fl": db_reading.tire_pressure_fl,
                "tire_pressure_fr": db_reading.tire_pressure_fr,
                "tire_pressure_rl": db_reading.tire_pressure_rl,
                "tire_pressure_rr": db_reading.tire_pressure_rr,
                "latitude": db_reading.latitude,
                "longitude": db_reading.longitude,
                "driving_mode": db_reading.driving_mode,
                "acceleration_g": db_reading.acceleration_g,
                "is_harsh_braking": db_reading.is_harsh_braking,
                "is_harsh_acceleration": db_reading.is_harsh_acceleration,
                "is_speeding": db_reading.is_speeding,
                "heading": db_reading.heading or 0
            }
        }
    )
    
    return db_reading


@router.post("/readings/batch")
async def create_telemetry_batch(
    batch: TelemetryBatchCreate,
    db: AsyncSession=Depends(get_db)
):
    """
    Ingest multiple telemetry readings at once (for efficiency)
    """
    if not batch.readings:
        return {"created": 0}
    
    # Verify vehicle exists
    vehicle_id = batch.readings[0].vehicle_id
    vehicle_result = await db.execute(
        select(Vehicle).where(Vehicle.id == vehicle_id)
    )
    if not vehicle_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    # Get active trip
    active_trip_result = await db.execute(
        select(Trip)
        .where(Trip.vehicle_id == vehicle_id)
        .where(Trip.is_active == True)
    )
    active_trip = active_trip_result.scalar_one_or_none()
    
    created_count = 0
    max_speed = None
    max_rpm = None
    first_fuel = None
    last_fuel = None
    total_speed = 0
    total_rpm = 0
    distance_increment = 0
    mode_counts = {'parked': 0, 'city': 0, 'highway': 0, 'sport': 0, 'reverse': 0}
    harsh_brake_count = 0
    harsh_accel_count = 0
    
    for reading in batch.readings:
        db_reading = TelemetryReading(
            vehicle_id=reading.vehicle_id,
            time=reading.time or utc_now(),
            speed_kmh=reading.speed_kmh,
            rpm=reading.rpm,
            gear=reading.gear,
            throttle_position=reading.throttle_position,
            engine_temp=reading.engine_temp,
            oil_temp=reading.oil_temp,
            oil_pressure=reading.oil_pressure,
            fuel_level=reading.fuel_level,
            acceleration_g=reading.acceleration_g,
            is_harsh_braking=reading.is_harsh_braking,
            is_harsh_acceleration=reading.is_harsh_acceleration,
            is_speeding=reading.is_speeding,
            is_idling=reading.is_idling,
            driving_mode=reading.driving_mode,
            engine_stress_score=reading.engine_stress_score
        )
        db.add(db_reading)
        created_count += 1
        
        # Track stats for trip update
        if max_speed is None or reading.speed_kmh > max_speed:
            max_speed = reading.speed_kmh
        if reading.rpm and (max_rpm is None or reading.rpm > max_rpm):
            max_rpm = reading.rpm
        if first_fuel is None and reading.fuel_level:
            first_fuel = reading.fuel_level
        if reading.fuel_level:
            last_fuel = reading.fuel_level
        
        # Accumulate for averages and distance
        total_speed += reading.speed_kmh
        total_rpm += reading.rpm or 0
        distance_increment += reading.speed_kmh / 3600.0  # 1 second per reading
        
        # Track driving modes
        if reading.driving_mode:
            mode = reading.driving_mode.lower()
            if mode in mode_counts:
                mode_counts[mode] += 1
        
        # Track harsh events
        if reading.is_harsh_braking:
            harsh_brake_count += 1
        if reading.is_harsh_acceleration:
            harsh_accel_count += 1
    
    # Update trip with batch stats
    if active_trip:
        active_trip.total_readings = (active_trip.total_readings or 0) + created_count
        
        # Update speed sum and distance
        active_trip.speed_sum = (active_trip.speed_sum or 0) + total_speed
        active_trip.rpm_sum = (active_trip.rpm_sum or 0) + total_rpm
        active_trip.distance_km = (active_trip.distance_km or 0) + distance_increment
        
        if max_speed and (active_trip.max_speed_kmh is None or max_speed > active_trip.max_speed_kmh):
            active_trip.max_speed_kmh = max_speed
        
        if max_rpm and (active_trip.max_rpm is None or max_rpm > active_trip.max_rpm):
            active_trip.max_rpm = max_rpm
        
        # Update driving mode counters
        active_trip.mode_parked_seconds = (active_trip.mode_parked_seconds or 0) + mode_counts['parked']
        active_trip.mode_city_seconds = (active_trip.mode_city_seconds or 0) + mode_counts['city']
        active_trip.mode_highway_seconds = (active_trip.mode_highway_seconds or 0) + mode_counts['highway']
        active_trip.mode_sport_seconds = (active_trip.mode_sport_seconds or 0) + mode_counts['sport']
        active_trip.mode_reverse_seconds = (active_trip.mode_reverse_seconds or 0) + mode_counts['reverse']
        
        # Update harsh event counts
        active_trip.harsh_brake_count = (active_trip.harsh_brake_count or 0) + harsh_brake_count
        active_trip.harsh_accel_count = (active_trip.harsh_accel_count or 0) + harsh_accel_count
        
        if active_trip.fuel_start is None and first_fuel:
            active_trip.fuel_start = first_fuel
        
        if last_fuel:
            active_trip.fuel_end = last_fuel
            if active_trip.fuel_start:
                active_trip.fuel_used_liters = active_trip.fuel_start - last_fuel
    
    await db.commit()
    
    return {"created": created_count}


@router.get("/readings/{vehicle_id}", response_model=List[TelemetryReadingResponse])
async def get_telemetry_history(
    vehicle_id: UUID,
    limit: int=Query(default=100, ge=1, le=1000),
    hours: int=Query(default=24, ge=1, le=168),
    db: AsyncSession=Depends(get_db)
):
    """
    Get telemetry reading history for a vehicle
    """
    since = utc_now() - timedelta(hours=hours)
    
    result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
        .where(TelemetryReading.time >= since)
        .order_by(TelemetryReading.time.desc())
        .limit(limit)
    )
    
    return result.scalars().all()

# ============================================
# WebSocket Endpoint
# ============================================


async def _handle_websocket(websocket: WebSocket, vehicle_id: str):
    """Common WebSocket handler for real-time telemetry streaming"""
    await manager.connect(websocket, vehicle_id)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Could handle client commands here if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket, vehicle_id)


@router.websocket("/ws/{vehicle_id}")
async def websocket_endpoint(websocket: WebSocket, vehicle_id: str):
    """WebSocket endpoint at /ws/{vehicle_id}"""
    await _handle_websocket(websocket, vehicle_id)


@router.websocket("/stream/{vehicle_id}")
async def websocket_stream_endpoint(websocket: WebSocket, vehicle_id: str):
    """WebSocket endpoint at /stream/{vehicle_id} (frontend compatibility)"""
    await _handle_websocket(websocket, vehicle_id)

# ============================================
# ML MODEL STATUS & TRAINING ENDPOINTS
# ============================================


# Check if ML modules are available
ML_AVAILABLE = False
try:
    from app.ml.training import behavior_classifier, anomaly_detector, data_collector
    from app.ml.features.extractor import FeatureExtractor
    from app.ml.models.driver_scorer import DriverScorer
    ML_AVAILABLE = True
except ImportError:
    pass


@router.get("/ml/status")
async def get_ml_status():
    """
    Get overall ML system status
    """
    if not ML_AVAILABLE:
        return {"ml_available": False, "error": "ML modules not installed"}
    
    return {
        "ml_available": True,
        "modules": {
            "feature_extractor": True,
            "behavior_classifier": True,
            "anomaly_detector": True,
            "driver_scorer": True
        }
    }


@router.get("/ml/models/status")
async def get_ml_models_status():
    """
    Get status of trained ML models
    """
    if not ML_AVAILABLE:
        return {"ml_available": False, "error": "ML service not available"}
    
    try:
        from pathlib import Path
        import json
        
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
                metadata_file = model_dir / f"{name}_metadata.json"
                
                model_info = {
                    "file": str(model_file),
                    "exists": True,
                }
                
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        model_info["metadata"] = json.load(f)
                
                status["models"][name] = model_info
        
        return status
        
    except ImportError as e:
        return {"ml_available": True, "sklearn_available": False, "error": str(e)}


@router.post("/ml/train/{vehicle_id}")
async def train_ml_models(
    vehicle_id: UUID,
    days: int=Query(default=30, ge=1, le=365),
    train_behavior: bool=Query(default=True),
    train_anomaly: bool=Query(default=True),
    db: AsyncSession=Depends(get_db)
):
    """
    Train ML models using historical trip data
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    since = utc_now() - timedelta(days=days)
    
    # Get completed trips
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
    
    # Get all readings
    all_readings_result = await db.execute(
        select(TelemetryReading)
        .where(TelemetryReading.vehicle_id == vehicle_id)
        .order_by(TelemetryReading.time)
    )
    all_readings_orm = all_readings_result.scalars().all()
    
    if not all_readings_orm:
        raise HTTPException(status_code=400, detail="No telemetry readings found")
    
    # Convert readings to feature format
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
    
    # Extract features and train
    feature_extractor = FeatureExtractor()
    scorer = DriverScorer()
    examples = []
    
    reading_offset = 0
    skipped_trips = 0
    
    for trip in trips:
        trip_reading_count = trip.total_readings or 0
        
        if trip_reading_count < 30:
            reading_offset += trip_reading_count
            skipped_trips += 1
            continue
        
        trip_readings = all_readings[reading_offset:reading_offset + trip_reading_count]
        reading_offset += trip_reading_count
        
        if len(trip_readings) < 30:
            skipped_trips += 1
            continue
        
        # Synthetic timestamps for proper duration calculation
        if trip_readings:
            from datetime import timedelta as td
            base_time = datetime.fromisoformat(
                trip_readings[0]["timestamp"].replace('Z', '+00:00')
            ) if trip_readings[0].get("timestamp") else utc_now()
            for i, reading in enumerate(trip_readings):
                reading["timestamp"] = (base_time + td(seconds=i)).isoformat()
        
        # Extract features
        features = feature_extractor.extract_from_readings(
            trip_readings, str(trip.id), str(vehicle_id)
        )
        
        if features is None:
            continue
        
        # Get rule-based score for labeling
        rule_score = scorer.score_trip(features)
        
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
            detail=f"Only {len(examples)} valid examples. Skipped {skipped_trips} trips."
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
    
    # Reset hybrid_scorer cache so next scoring will load fresh models
    try:
        from app.ml.models.hybrid_scorer import hybrid_scorer
        hybrid_scorer._loaded_vehicle_id = None  # Force reload on next use
    except ImportError:
        pass
    
    return results


@router.get("/debug/db")
async def debug_database(db: AsyncSession=Depends(get_db)):
    """
    Debug endpoint to check database connectivity
    """
    try:
        result = await db.execute(text("SELECT 1"))
        return {"database": "connected", "result": result.scalar()}
    except Exception as e:
        return {"database": "error", "error": str(e)}

# ============================================
# BACKWARD COMPATIBILITY ROUTES
# These redirect to the new modular endpoints
# TODO: Update frontend to use new paths, then remove these
# ============================================


from fastapi.responses import RedirectResponse


@router.get("/vehicles")
async def get_vehicles_compat(db: AsyncSession=Depends(get_db)):
    """[COMPAT] Get vehicles - redirects to /api/vehicles"""
    from app.api.vehicles import get_vehicles
    return await get_vehicles(db)


@router.post("/vehicles")
async def create_vehicle_compat(vehicle: dict, db: AsyncSession=Depends(get_db)):
    """[COMPAT] Create vehicle - redirects to /api/vehicles"""
    from app.api.vehicles import register_vehicle
    from app.schemas.telemetry import VehicleCreate
    return await register_vehicle(VehicleCreate(**vehicle), db)


@router.get("/trips/{vehicle_id}")
async def get_trips_compat(vehicle_id: UUID, limit: int=10, db: AsyncSession=Depends(get_db)):
    """[COMPAT] Get trips - redirects to /api/trips/{vehicle_id}"""
    from app.api.trips import get_trips
    return await get_trips(vehicle_id, limit, False, db)


@router.get("/trips/active/{vehicle_id}")
async def get_active_trip_compat(vehicle_id: UUID, db: AsyncSession=Depends(get_db)):
    """[COMPAT] Get active trip - redirects to /api/trips/active/{vehicle_id}"""
    from app.api.trips import get_active_trip
    return await get_active_trip(vehicle_id, db)


@router.post("/trips/start/{vehicle_id}")
async def start_trip_compat(vehicle_id: UUID, db: AsyncSession=Depends(get_db)):
    """[COMPAT] Start trip - redirects to /api/trips/start/{vehicle_id}"""
    from app.api.trips import start_trip
    return await start_trip(vehicle_id, db)


# Frontend sends vehicle_id in body, not URL
class TripStartRequest(BaseModel):
    vehicle_id: UUID
    start_latitude: Optional[float] = None
    start_longitude: Optional[float] = None


class TripEndRequest(BaseModel):
    end_latitude: Optional[float] = None
    end_longitude: Optional[float] = None


@router.post("/trips/start")
async def start_trip_body_compat(request: TripStartRequest, db: AsyncSession=Depends(get_db)):
    """[COMPAT] Start trip with vehicle_id in body"""
    from app.api.trips import start_trip
    return await start_trip(request.vehicle_id, db)


@router.post("/trips/{trip_id}/end")
async def end_trip_path_compat(trip_id: UUID, db: AsyncSession=Depends(get_db)):
    """[COMPAT] End trip - /trips/{trip_id}/end format"""
    from app.api.trips import end_trip
    return await end_trip(trip_id, db)


@router.post("/trips/end-active/{vehicle_id}")
async def end_active_trip_compat(vehicle_id: UUID, db: AsyncSession=Depends(get_db)):
    """[COMPAT] End active trip for a vehicle"""
    from app.api.trips import get_active_trip, end_trip
    active = await get_active_trip(vehicle_id, db)
    if active:
        return await end_trip(active.id, db)
    return None


@router.post("/trips/end/{trip_id}")
async def end_trip_compat(trip_id: UUID, db: AsyncSession=Depends(get_db)):
    """[COMPAT] End trip - redirects to /api/trips/end/{trip_id}"""
    from app.api.trips import end_trip
    return await end_trip(trip_id, db)


@router.get("/stats/weekly/{vehicle_id}")
async def get_weekly_stats_compat(vehicle_id: UUID, db: AsyncSession=Depends(get_db)):
    """[COMPAT] Get weekly stats - redirects to /api/analytics/stats/weekly/{vehicle_id}"""
    from app.api.analytics import get_weekly_stats
    return await get_weekly_stats(vehicle_id, db)


@router.get("/ml/summary/{vehicle_id}")
async def get_ml_summary_compat(vehicle_id: UUID, days: int=7, db: AsyncSession=Depends(get_db)):
    """[COMPAT] Get ML summary - redirects to /api/scoring/summary/{vehicle_id}"""
    from app.api.scoring import get_driver_summary
    return await get_driver_summary(str(vehicle_id), days, db)
