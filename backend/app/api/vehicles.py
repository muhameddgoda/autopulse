"""
Vehicle management endpoints
Handles vehicle CRUD operations
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from app.database import get_db
from app.models.telemetry import Vehicle
from app.schemas.telemetry import VehicleCreate, VehicleResponse

router = APIRouter()


@router.post("", response_model=VehicleResponse)
async def register_vehicle(
    vehicle: VehicleCreate,
    db: AsyncSession=Depends(get_db)
):
    """
    Register a new vehicle in the system
    """
    db_vehicle = Vehicle(**vehicle.dict())
    db.add(db_vehicle)
    await db.commit()
    await db.refresh(db_vehicle)
    return db_vehicle


@router.get("", response_model=list[VehicleResponse])
async def get_vehicles(db: AsyncSession=Depends(get_db)):
    """
    Get all registered vehicles
    """
    result = await db.execute(select(Vehicle))
    return result.scalars().all()


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(
    vehicle_id: UUID,
    db: AsyncSession=Depends(get_db)
):
    """
    Get a specific vehicle by ID
    """
    result = await db.execute(
        select(Vehicle).where(Vehicle.id == vehicle_id)
    )
    vehicle = result.scalar_one_or_none()
    
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    return vehicle


@router.delete("/{vehicle_id}")
async def delete_vehicle(
    vehicle_id: UUID,
    db: AsyncSession=Depends(get_db)
):
    """
    Delete a vehicle (soft delete could be implemented)
    """
    result = await db.execute(
        select(Vehicle).where(Vehicle.id == vehicle_id)
    )
    vehicle = result.scalar_one_or_none()
    
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    await db.delete(vehicle)
    await db.commit()
    
    return {"status": "deleted", "vehicle_id": str(vehicle_id)}
