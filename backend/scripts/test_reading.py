#!/usr/bin/env python3
"""
Quick test to verify telemetry readings are working
"""
import asyncio
import httpx

API_BASE = "http://localhost:8000/api/telemetry"

async def test_reading():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get vehicle
        resp = await client.get(f"{API_BASE}/vehicles")
        print(f"Vehicles response: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            return
        
        vehicles = resp.json()
        if not vehicles:
            print("No vehicles found!")
            return
        
        vehicle_id = vehicles[0]["id"]
        print(f"Using vehicle: {vehicle_id}")
        
        # Start a trip
        trip_resp = await client.post(
            f"{API_BASE}/trips/start",
            json={
                "vehicle_id": vehicle_id,
                "start_latitude": 48.7758,
                "start_longitude": 9.1829
            }
        )
        print(f"Start trip response: {trip_resp.status_code}")
        if trip_resp.status_code != 200:
            print(f"Error: {trip_resp.text}")
            return
        
        trip_data = trip_resp.json()
        trip_id = trip_data["id"]
        print(f"Trip ID: {trip_id}")
        
        # Send a test reading
        reading = {
            "vehicle_id": vehicle_id,
            "speed_kmh": 120.5,
            "rpm": 4500,
            "gear": 5,
            "throttle_position": 65.0,
            "engine_temp": 92.0,
            "oil_temp": 105.0,
            "oil_pressure": 3.5,
            "fuel_level": 75.0,
            "battery_voltage": 14.2,
            "tire_pressure_fl": 33.0,
            "tire_pressure_fr": 33.0,
            "tire_pressure_rl": 32.5,
            "tire_pressure_rr": 32.5,
            "latitude": 48.7758,
            "longitude": 9.1829,
            "heading": 90.0,
            "driving_mode": "highway",
            "acceleration_g": 0.15,
            "jerk_ms3": 0.5,
            "is_harsh_braking": False,
            "is_harsh_acceleration": False,
            "is_over_rpm": False,
            "is_speeding": False,
            "is_idling": False,
            "engine_stress_score": 35.0
        }
        
        reading_resp = await client.post(f"{API_BASE}/reading", json=reading)
        print(f"Reading response: {reading_resp.status_code}")
        if reading_resp.status_code != 200:
            print(f"Error: {reading_resp.text}")
        else:
            print(f"Reading saved: {reading_resp.json()}")
        
        # Send a few more readings
        for i in range(5):
            reading["speed_kmh"] = 100 + i * 10
            reading["rpm"] = 3500 + i * 200
            resp = await client.post(f"{API_BASE}/reading", json=reading)
            print(f"Reading {i+1}: {resp.status_code}")
        
        # End trip
        end_resp = await client.post(
            f"{API_BASE}/trips/{trip_id}/end",
            json={
                "end_latitude": 48.78,
                "end_longitude": 9.19
            }
        )
        print(f"End trip response: {end_resp.status_code}")
        if end_resp.status_code == 200:
            end_data = end_resp.json()
            print(f"Trip ended:")
            print(f"  avg_speed_kmh: {end_data.get('avg_speed_kmh')}")
            print(f"  max_speed_kmh: {end_data.get('max_speed_kmh')}")
            print(f"  distance_km: {end_data.get('distance_km')}")
            print(f"  total_readings: {end_data.get('total_readings')}")
        else:
            print(f"Error: {end_resp.text}")

if __name__ == "__main__":
    asyncio.run(test_reading())
