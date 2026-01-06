#!/usr/bin/env python3
"""
AutoPulse Training Data Generator
=================================
Generates diverse driving scenarios for ML model training.

This script creates trips with varied driving behaviors:
- Exemplary (very safe, smooth driving)
- Calm (safe, defensive driving)
- Normal (average driving patterns)
- Aggressive (risky, harsh events)
- Dangerous (very dangerous, many violations)
- Anomalous (unusual patterns for anomaly detection)

Usage:
    python generate_training_data.py --trips 200 --vehicle-id <uuid>
    python generate_training_data.py --trips 200 --create-vehicle
"""

import argparse
import asyncio
import random
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import httpx

# API Configuration
API_BASE = "http://localhost:8000/api/telemetry"

@dataclass
class DrivingProfile:
    """Defines characteristics for a driving behavior type"""
    name: str
    
    # Speed behavior
    base_speed_range: Tuple[float, float]  # km/h
    speed_variance: float
    speeding_probability: float  # 0-1
    max_speed_factor: float  # multiplier for speed limit
    
    # Acceleration behavior
    acceleration_range: Tuple[float, float]  # g
    harsh_accel_probability: float
    harsh_brake_probability: float
    max_accel_g: float
    max_decel_g: float
    
    # RPM behavior
    rpm_range: Tuple[int, int]
    over_rpm_probability: float
    
    # Throttle behavior
    throttle_range: Tuple[float, float]
    throttle_variance: float
    
    # Mode preferences
    sport_mode_probability: float
    
    # Trip characteristics
    duration_range: Tuple[int, int]  # seconds
    idle_probability: float
    
    # Engine stress
    stress_base: float
    stress_variance: float


# Define driving profiles
PROFILES: Dict[str, DrivingProfile] = {
    "exemplary": DrivingProfile(
        name="exemplary",
        base_speed_range=(40, 100),
        speed_variance=5,
        speeding_probability=0.02,
        max_speed_factor=1.05,
        acceleration_range=(0.05, 0.2),
        harsh_accel_probability=0.01,
        harsh_brake_probability=0.02,
        max_accel_g=0.25,
        max_decel_g=0.3,
        rpm_range=(1200, 2800),
        over_rpm_probability=0.0,
        throttle_range=(10, 40),
        throttle_variance=5,
        sport_mode_probability=0.0,
        duration_range=(180, 400),  # 3-6 minutes
        idle_probability=0.05,
        stress_base=15,
        stress_variance=5,
    ),
    "calm": DrivingProfile(
        name="calm",
        base_speed_range=(50, 120),
        speed_variance=10,
        speeding_probability=0.1,
        max_speed_factor=1.1,
        acceleration_range=(0.1, 0.3),
        harsh_accel_probability=0.05,
        harsh_brake_probability=0.08,
        max_accel_g=0.35,
        max_decel_g=0.45,
        rpm_range=(1500, 3500),
        over_rpm_probability=0.02,
        throttle_range=(15, 50),
        throttle_variance=8,
        sport_mode_probability=0.1,
        duration_range=(150, 350),  # 2.5-6 minutes
        idle_probability=0.08,
        stress_base=25,
        stress_variance=8,
    ),
    "normal": DrivingProfile(
        name="normal",
        base_speed_range=(60, 140),
        speed_variance=15,
        speeding_probability=0.25,
        max_speed_factor=1.2,
        acceleration_range=(0.15, 0.45),
        harsh_accel_probability=0.15,
        harsh_brake_probability=0.15,
        max_accel_g=0.5,
        max_decel_g=0.6,
        rpm_range=(1800, 4500),
        over_rpm_probability=0.08,
        throttle_range=(20, 65),
        throttle_variance=12,
        sport_mode_probability=0.25,
        duration_range=(120, 300),  # 2-5 minutes
        idle_probability=0.1,
        stress_base=35,
        stress_variance=12,
    ),
    "aggressive": DrivingProfile(
        name="aggressive",
        base_speed_range=(80, 180),
        speed_variance=25,
        speeding_probability=0.5,
        max_speed_factor=1.4,
        acceleration_range=(0.3, 0.7),
        harsh_accel_probability=0.35,
        harsh_brake_probability=0.35,
        max_accel_g=0.85,
        max_decel_g=1.2,
        rpm_range=(2500, 5500),
        over_rpm_probability=0.2,
        throttle_range=(35, 85),
        throttle_variance=18,
        sport_mode_probability=0.6,
        duration_range=(100, 250),  # 1.5-4 minutes
        idle_probability=0.05,
        stress_base=55,
        stress_variance=15,
    ),
    "dangerous": DrivingProfile(
        name="dangerous",
        base_speed_range=(100, 220),
        speed_variance=35,
        speeding_probability=0.75,
        max_speed_factor=1.6,
        acceleration_range=(0.5, 1.0),
        harsh_accel_probability=0.55,
        harsh_brake_probability=0.55,
        max_accel_g=1.1,
        max_decel_g=1.8,
        rpm_range=(3500, 6500),
        over_rpm_probability=0.4,
        throttle_range=(50, 100),
        throttle_variance=25,
        sport_mode_probability=0.85,
        duration_range=(80, 200),  # 1-3 minutes
        idle_probability=0.02,
        stress_base=75,
        stress_variance=20,
    ),
    "anomalous_slow": DrivingProfile(
        name="anomalous_slow",
        base_speed_range=(10, 40),  # Very slow
        speed_variance=5,
        speeding_probability=0.0,
        max_speed_factor=1.0,
        acceleration_range=(0.02, 0.1),
        harsh_accel_probability=0.0,
        harsh_brake_probability=0.4,  # Sudden stops
        max_accel_g=0.15,
        max_decel_g=0.8,
        rpm_range=(800, 1800),
        over_rpm_probability=0.0,
        throttle_range=(5, 25),
        throttle_variance=3,
        sport_mode_probability=0.0,
        duration_range=(200, 400),  # 3-6 minutes
        idle_probability=0.4,  # Lots of idling
        stress_base=10,
        stress_variance=5,
    ),
    "anomalous_erratic": DrivingProfile(
        name="anomalous_erratic",
        base_speed_range=(30, 200),  # Wild swings
        speed_variance=50,
        speeding_probability=0.4,
        max_speed_factor=1.5,
        acceleration_range=(0.1, 0.9),  # Inconsistent
        harsh_accel_probability=0.6,
        harsh_brake_probability=0.6,
        max_accel_g=1.0,
        max_decel_g=1.5,
        rpm_range=(1000, 6000),  # Wide range
        over_rpm_probability=0.3,
        throttle_range=(5, 100),  # Full range
        throttle_variance=40,
        sport_mode_probability=0.5,
        duration_range=(120, 240),  # 2-4 minutes
        idle_probability=0.2,
        stress_base=50,
        stress_variance=30,
    ),
}

# Distribution of trips by behavior type (total = 200)
TRIP_DISTRIBUTION = {
    "exemplary": 25,
    "calm": 40,
    "normal": 50,
    "aggressive": 40,
    "dangerous": 25,
    "anomalous_slow": 10,
    "anomalous_erratic": 10,
}


class TelemetryGenerator:
    """Generates realistic telemetry data based on driving profiles"""
    
    def __init__(self, profile: DrivingProfile):
        self.profile = profile
        self.current_speed = 0.0
        self.current_rpm = 1000
        self.current_throttle = 0.0
        self.current_gear = 1
        self.engine_temp = 90.0
        self.oil_temp = 100.0
        self.fuel_level = random.uniform(50, 95)
        
        # Position (Stuttgart area)
        self.lat = 48.7758 + random.uniform(-0.05, 0.05)
        self.lon = 9.1829 + random.uniform(-0.05, 0.05)
        self.heading = random.uniform(0, 360)
        
        # Tracking
        self.harsh_brake_count = 0
        self.harsh_accel_count = 0
        self.speeding_count = 0
        self.idle_count = 0
        self.over_rpm_count = 0
        
    def generate_reading(self, elapsed_seconds: int) -> Dict:
        """Generate a single telemetry reading"""
        p = self.profile
        
        # Determine driving mode
        if random.random() < p.idle_probability and self.current_speed < 10:
            mode = "parked"
            target_speed = 0
        elif random.random() < p.sport_mode_probability:
            mode = "sport"
            target_speed = random.uniform(*p.base_speed_range) * 1.2
        elif self.current_speed > 100:
            mode = "highway"
            target_speed = random.uniform(100, p.base_speed_range[1])
        else:
            mode = "city"
            target_speed = random.uniform(p.base_speed_range[0], min(80, p.base_speed_range[1]))
        
        # Add variance
        target_speed += random.gauss(0, p.speed_variance)
        target_speed = max(0, min(250, target_speed))
        
        # Calculate acceleration
        speed_diff = target_speed - self.current_speed
        
        # Determine if harsh event
        is_harsh_braking = False
        is_harsh_acceleration = False
        
        if speed_diff < -10 and random.random() < p.harsh_brake_probability:
            # Harsh braking
            accel_g = -random.uniform(0.4, p.max_decel_g)
            is_harsh_braking = True
            self.harsh_brake_count += 1
        elif speed_diff > 10 and random.random() < p.harsh_accel_probability:
            # Harsh acceleration
            accel_g = random.uniform(0.35, p.max_accel_g)
            is_harsh_acceleration = True
            self.harsh_accel_count += 1
        else:
            # Normal acceleration
            accel_g = random.uniform(*p.acceleration_range) * (1 if speed_diff > 0 else -1)
        
        # Apply acceleration (convert g to speed change per second)
        speed_change = accel_g * 9.81 * 3.6  # m/s² to km/h/s
        self.current_speed = max(0, min(250, self.current_speed + speed_change))
        
        # Determine if speeding
        speed_limit = 130 if mode == "highway" else 50
        is_speeding = self.current_speed > speed_limit
        if is_speeding:
            self.speeding_count += 1
        
        # Is idling
        is_idling = self.current_speed < 5 and mode == "parked"
        if is_idling:
            self.idle_count += 1
        
        # Calculate gear based on speed
        if self.current_speed < 15:
            self.current_gear = 1
        elif self.current_speed < 30:
            self.current_gear = 2
        elif self.current_speed < 50:
            self.current_gear = 3
        elif self.current_speed < 80:
            self.current_gear = 4
        elif self.current_speed < 120:
            self.current_gear = 5
        else:
            self.current_gear = 6
        
        # Calculate RPM
        base_rpm = p.rpm_range[0] + (self.current_speed / 250) * (p.rpm_range[1] - p.rpm_range[0])
        rpm_variance = random.gauss(0, 200)
        self.current_rpm = int(max(800, min(7000, base_rpm + rpm_variance)))
        
        # Check over-RPM
        is_over_rpm = self.current_rpm > 5500
        if is_over_rpm:
            self.over_rpm_count += 1
        
        # Throttle
        target_throttle = (self.current_speed / 200) * 100 + random.gauss(0, p.throttle_variance)
        if is_harsh_acceleration:
            target_throttle = min(100, target_throttle + 30)
        self.current_throttle = max(0, min(100, target_throttle))
        
        # Engine metrics
        self.engine_temp = 90 + (self.current_rpm / 7000) * 20 + random.gauss(0, 2)
        self.oil_temp = 100 + (self.current_rpm / 7000) * 15 + random.gauss(0, 2)
        oil_pressure = 2.0 + (self.current_rpm / 7000) * 3 + random.gauss(0, 0.2)
        
        # Fuel consumption
        fuel_rate = 0.0001 * (self.current_rpm / 2000) * (self.current_throttle / 50)
        self.fuel_level = max(5, self.fuel_level - fuel_rate)
        
        # Update position
        distance_km = (self.current_speed / 3600)  # km per second
        heading_rad = math.radians(self.heading)
        self.lat += distance_km * math.cos(heading_rad) / 111
        self.lon += distance_km * math.sin(heading_rad) / (111 * math.cos(math.radians(self.lat)))
        self.heading += random.gauss(0, 5)
        self.heading = self.heading % 360
        
        # Tire pressures (slight variations)
        tire_base = 33 + random.gauss(0, 0.5)
        
        # Engine stress
        stress = p.stress_base + random.gauss(0, p.stress_variance)
        if is_harsh_acceleration or is_harsh_braking:
            stress += 15
        if is_speeding:
            stress += 10
        stress = max(0, min(100, stress))
        
        # Jerk (rate of acceleration change)
        jerk = random.gauss(0, 2) if not (is_harsh_acceleration or is_harsh_braking) else random.uniform(5, 15)
        
        # NOTE: No 'timestamp' field - the backend sets the time
        return {
            "speed_kmh": round(self.current_speed, 1),
            "rpm": self.current_rpm,
            "gear": self.current_gear,
            "throttle_position": round(self.current_throttle, 1),
            "engine_temp": round(self.engine_temp, 1),
            "oil_temp": round(self.oil_temp, 1),
            "oil_pressure": round(oil_pressure, 2),
            "fuel_level": round(self.fuel_level, 1),
            "battery_voltage": round(14.2 + random.gauss(0, 0.1), 1),
            "tire_pressure_fl": round(tire_base + random.gauss(0, 0.3), 1),
            "tire_pressure_fr": round(tire_base + random.gauss(0, 0.3), 1),
            "tire_pressure_rl": round(tire_base - 0.5 + random.gauss(0, 0.3), 1),
            "tire_pressure_rr": round(tire_base - 0.5 + random.gauss(0, 0.3), 1),
            "latitude": round(self.lat, 6),
            "longitude": round(self.lon, 6),
            "heading": round(self.heading, 1),
            "driving_mode": mode,
            "acceleration_g": round(accel_g, 3),
            "jerk_ms3": round(jerk, 2),
            "is_harsh_braking": is_harsh_braking,
            "is_harsh_acceleration": is_harsh_acceleration,
            "is_over_rpm": is_over_rpm,
            "is_speeding": is_speeding,
            "is_idling": is_idling,
            "engine_stress_score": round(stress, 1),
        }


async def create_vehicle(client: httpx.AsyncClient) -> str:
    """Create a new vehicle for testing"""
    vehicle_data = {
        "vin": f"WP0ZZZ99Z{random.randint(10000000, 99999999)}",
        "make": "Porsche",
        "model": "911",
        "variant": "Carrera S",
        "year": 2024,
        "color": "Guards Red"
    }
    
    response = await client.post(f"{API_BASE}/vehicles", json=vehicle_data)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Created vehicle: {data['id']}")
        return data['id']
    else:
        raise Exception(f"Failed to create vehicle: {response.text}")


async def generate_trip(
    client: httpx.AsyncClient,
    vehicle_id: str,
    profile: DrivingProfile,
    trip_number: int,
    total_trips: int,
    days_back: int = 0
) -> bool:
    """Generate a complete trip with telemetry data"""
    
    # Trip duration
    duration = random.randint(*profile.duration_range)
    # Generate number of readings equal to duration (1 reading = 1 second)
    # This is the KEY: backend uses max(wall_clock, total_readings) for duration
    readings_count = duration  # Match readings to simulated seconds!
    
    print(f"  [{trip_number}/{total_trips}] Generating {profile.name} trip ({duration}s, {readings_count} readings)...")
    
    generator = TelemetryGenerator(profile)
    
    # Start trip
    try:
        start_response = await client.post(
            f"{API_BASE}/trips/start",
            json={
                "vehicle_id": vehicle_id,
                "start_latitude": generator.lat,
                "start_longitude": generator.lon
            }
        )
    except Exception as e:
        print(f"    ❌ Connection error starting trip: {e}")
        return False
    
    if start_response.status_code != 200:
        print(f"    ❌ Failed to start trip: {start_response.text}")
        return False
    
    trip_data = start_response.json()
    trip_id = trip_data["id"]
    
    # Generate and send readings
    successful_readings = 0
    failed_readings = 0
    
    for i in range(readings_count):
        reading = generator.generate_reading(i)
        reading["vehicle_id"] = vehicle_id
        
        try:
            resp = await client.post(f"{API_BASE}/reading", json=reading)
            if resp.status_code == 200:
                successful_readings += 1
            else:
                failed_readings += 1
                if failed_readings <= 2:
                    print(f"    ⚠️ Reading failed: {resp.status_code} - {resp.text[:100]}")
        except Exception as e:
            failed_readings += 1
            if failed_readings <= 2:
                print(f"    ⚠️ Reading error: {e}")
    
    # End trip
    try:
        end_response = await client.post(
            f"{API_BASE}/trips/{trip_id}/end",
            json={
                "end_latitude": generator.lat,
                "end_longitude": generator.lon
            }
        )
    except Exception as e:
        print(f"    ❌ Connection error ending trip: {e}")
        return False
    
    if end_response.status_code == 200:
        end_data = end_response.json()
        avg_speed = end_data.get('avg_speed_kmh', 0) or 0
        max_speed = end_data.get('max_speed_kmh', 0) or 0
        distance = end_data.get('distance_km', 0) or 0
        actual_duration = end_data.get('duration_seconds', 0) or 0
        
        # Calculate expected distance for verification
        expected_distance = (avg_speed * duration) / 3600
        
        status = "✅" if successful_readings >= 30 else "⚠️"
        print(f"    {status} {profile.name}: {distance:.1f}km, "
              f"duration={actual_duration}s, "
              f"avg={avg_speed:.0f}km/h, max={max_speed:.0f}km/h, "
              f"readings={successful_readings}/{readings_count}, "
              f"harsh_b={generator.harsh_brake_count}, harsh_a={generator.harsh_accel_count}")
        return successful_readings >= 30
    else:
        print(f"    ❌ Failed to end trip: {end_response.text}")
        return False


async def generate_all_trips(vehicle_id: str, total_trips: int = 200):
    """Generate all trips with diverse behaviors"""
    
    print(f"\n{'='*60}")
    print(f"🚗 AutoPulse Training Data Generator")
    print(f"{'='*60}")
    print(f"Vehicle ID: {vehicle_id}")
    print(f"Total trips to generate: {total_trips}")
    print(f"{'='*60}\n")
    
    # Calculate distribution
    distribution = {}
    remaining = total_trips
    
    total_weight = sum(TRIP_DISTRIBUTION.values())
    for behavior, count in TRIP_DISTRIBUTION.items():
        distribution[behavior] = int((count / total_weight) * total_trips)
        remaining -= distribution[behavior]
    
    # Add remaining to 'normal'
    distribution["normal"] += remaining
    
    print("Trip distribution:")
    for behavior, count in distribution.items():
        print(f"  • {behavior}: {count} trips")
    print()
    
    # Create trip list
    trips_to_generate = []
    for behavior, count in distribution.items():
        for _ in range(count):
            trips_to_generate.append(PROFILES[behavior])
    
    # Shuffle for variety
    random.shuffle(trips_to_generate)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        successful = 0
        failed = 0
        
        for i, profile in enumerate(trips_to_generate, 1):
            try:
                success = await generate_trip(client, vehicle_id, profile, i, total_trips)
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # Small delay between trips
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"  ❌ Error generating trip: {e}")
                failed += 1
            
            # Progress update every 20 trips
            if i % 20 == 0:
                print(f"\n📊 Progress: {i}/{total_trips} ({successful} successful, {failed} failed)\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Generation Complete!")
    print(f"{'='*60}")
    print(f"Successful trips: {successful}")
    print(f"Failed trips: {failed}")
    print(f"\nNext steps:")
    print(f"1. Train ML models:")
    print(f"   curl -X POST 'http://localhost:8000/api/telemetry/ml/train/{vehicle_id}?days=30'")
    print(f"\n2. Compare models (see model_comparison.py)")
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description="Generate training data for AutoPulse ML models")
    parser.add_argument("--trips", type=int, default=200, help="Number of trips to generate")
    parser.add_argument("--vehicle-id", type=str, help="Existing vehicle UUID")
    parser.add_argument("--create-vehicle", action="store_true", help="Create a new vehicle")
    
    args = parser.parse_args()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        if args.create_vehicle:
            vehicle_id = await create_vehicle(client)
        elif args.vehicle_id:
            # Validate the vehicle exists
            response = await client.get(f"{API_BASE}/vehicles/{args.vehicle_id}")
            if response.status_code == 200:
                vehicle_id = args.vehicle_id
                vehicle_data = response.json()
                print(f"Using vehicle: {vehicle_data.get('make', '')} {vehicle_data.get('model', '')} ({vehicle_id})")
            else:
                print(f"❌ Vehicle not found: {args.vehicle_id}")
                print("Fetching available vehicles...")
                response = await client.get(f"{API_BASE}/vehicles")
                if response.status_code == 200:
                    vehicles = response.json()
                    if vehicles:
                        print("\nAvailable vehicles:")
                        for v in vehicles:
                            print(f"  • {v['id']} - {v.get('make', '')} {v.get('model', '')}")
                        vehicle_id = vehicles[0]["id"]
                        print(f"\nUsing first vehicle: {vehicle_id}")
                    else:
                        print("No vehicles found. Creating one...")
                        vehicle_id = await create_vehicle(client)
                else:
                    raise Exception("Could not fetch vehicles")
        else:
            # Try to get existing vehicle
            response = await client.get(f"{API_BASE}/vehicles")
            if response.status_code == 200:
                vehicles = response.json()
                if vehicles:
                    vehicle_id = vehicles[0]["id"]
                    vehicle_data = vehicles[0]
                    print(f"Using existing vehicle: {vehicle_data.get('make', '')} {vehicle_data.get('model', '')} ({vehicle_id})")
                else:
                    vehicle_id = await create_vehicle(client)
            else:
                raise Exception("Could not fetch vehicles and no vehicle-id provided")
    
    await generate_all_trips(vehicle_id, args.trips)


if __name__ == "__main__":
    asyncio.run(main())