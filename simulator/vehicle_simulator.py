"""
AutoPulse Vehicle Simulator v4 - ML Enhanced
Porsche 911 Carrera S with full keyboard control + ML-ready metrics

NEW FEATURES:
- Acceleration tracking (m/s²)
- Harsh braking/acceleration event detection
- Idle time tracking
- Over-RPM event counting
- Jerk calculation (rate of acceleration change)
- Engine stress scoring

Controls:
  [1] City Mode
  [2] Highway Mode  
  [3] Sport Mode
  [P] Park
  [R] Reverse (only when stopped)
  [F] Low Fuel Test
  [T] High Temp Test
  [Q] Quit
"""

import asyncio
import random
import math
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import httpx

# For keyboard input on Windows
if os.name == 'nt':
    import msvcrt


class DrivingMode(Enum):
    PARKED = "parked"
    REVERSE = "reverse"
    CITY = "city"
    HIGHWAY = "highway"
    SPORT = "sport"


# Mode configurations
MODE_CONFIG = {
    DrivingMode.PARKED: {
        "min_speed": 0,
        "max_speed": 0,
        "target_speed_range": (0, 0),
        "acceleration": 0,
        "deceleration": 20,
        "stop_probability": 1.0,
        "color": "\033[90m",    # Gray
        "name": "PARKED"
    },
    DrivingMode.REVERSE: {
        "min_speed": 0,
        "max_speed": 15,
        "target_speed_range": (5, 10),
        "acceleration": 3,
        "deceleration": 10,
        "stop_probability": 0.0,
        "color": "\033[95m",    # Magenta
        "name": "REVERSE"
    },
    DrivingMode.CITY: {
        "min_speed": 0,
        "max_speed": 60,
        "target_speed_range": (25, 50),
        "acceleration": 10,
        "deceleration": 15,
        "stop_probability": 0.06,
        "color": "\033[93m",    # Yellow
        "name": "CITY"
    },
    DrivingMode.HIGHWAY: {
        "min_speed": 80,
        "max_speed": 160,
        "target_speed_range": (110, 140),
        "acceleration": 8,
        "deceleration": 10,
        "stop_probability": 0.0,
        "color": "\033[94m",    # Blue
        "name": "HIGHWAY"
    },
    DrivingMode.SPORT: {
        "min_speed": 0,
        "max_speed": 280,
        "target_speed_range": (140, 220),
        "acceleration": 30,
        "deceleration": 40,
        "stop_probability": 0.02,
        "color": "\033[91m",    # Red
        "name": "SPORT"
    },
}

# ML Thresholds for event detection
ML_THRESHOLDS = {
    "harsh_brake_g": -0.4,      # Deceleration > 0.4g is harsh braking
    "harsh_accel_g": 0.35,       # Acceleration > 0.35g is harsh acceleration
    "over_rpm": 6500,            # RPM above this is "over-revving"
    "redline_rpm": 7200,         # Redline territory
    "high_engine_temp": 105,     # °C - warning threshold
    "critical_engine_temp": 115, # °C - critical threshold
    "high_throttle": 90,         # % - aggressive throttle
    "idle_speed": 5,             # km/h - below this is considered idle
    "speeding_threshold": 130,   # km/h - general speeding
}


@dataclass
class MLMetrics:
    """Machine Learning ready derived metrics"""
    # Acceleration metrics
    acceleration_ms2: float = 0.0          # Current acceleration in m/s²
    acceleration_g: float = 0.0            # Current acceleration in G-force
    jerk_ms3: float = 0.0                  # Rate of acceleration change
    
    # Event counters (per trip/session)
    harsh_brake_count: int = 0             # Count of harsh braking events
    harsh_accel_count: int = 0             # Count of harsh acceleration events
    over_rpm_seconds: int = 0              # Seconds spent over RPM threshold
    redline_seconds: int = 0               # Seconds in redline
    over_speed_seconds: int = 0            # Seconds speeding
    high_throttle_seconds: int = 0         # Seconds at high throttle
    
    # Time tracking
    idle_seconds: int = 0                  # Total idle time (engine on, not moving)
    moving_seconds: int = 0                # Total moving time
    
    # Running averages (for trip summary)
    avg_acceleration_g: float = 0.0
    max_acceleration_g: float = 0.0
    max_deceleration_g: float = 0.0        # Most negative (stored as positive)
    
    # Engine stress
    engine_stress_score: float = 0.0       # 0-100 composite score
    
    # Previous values for derivative calculations
    _prev_speed_ms: float = 0.0
    _prev_acceleration: float = 0.0
    _acceleration_samples: int = 0
    _acceleration_sum: float = 0.0
    
    # Event flags (for detecting edge transitions)
    _in_harsh_brake: bool = False
    _in_harsh_accel: bool = False


@dataclass
class VehicleState:
    speed_kmh: float = 0.0
    target_speed: float = 0.0
    rpm: int = 800
    gear: int = 0
    throttle_position: float = 0.0
    
    engine_temp: float = 20.0
    oil_temp: float = 20.0
    oil_pressure: float = 1.0
    
    fuel_level: float = 85.0
    battery_voltage: float = 12.6
    
    # Tire pressure (PSI)
    tire_pressure_fl: float = 33.0
    tire_pressure_fr: float = 33.0
    tire_pressure_rl: float = 32.0
    tire_pressure_rr: float = 32.0
    
    latitude: float = 48.8342
    longitude: float = 9.1519
    heading: float = 0.0
    
    mode: DrivingMode = DrivingMode.PARKED
    engine_running: bool = False
    trip_active: bool = False
    trip_id: str = None
    
    # ML Metrics
    ml: MLMetrics = field(default_factory=MLMetrics)
    
    MAX_RPM: int = 7500
    IDLE_RPM: int = 800
    
    GEAR_SPEEDS: dict = field(default_factory=lambda: {
        1: (0, 45),
        2: (20, 75),
        3: (40, 110),
        4: (60, 150),
        5: (80, 190),
        6: (100, 240),
        7: (120, 320),
    })


class Porsche911Simulator:
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        vehicle_id: str = None,
        interval_ms: int = 1000
    ):
        self.api_url = api_url
        self.vehicle_id = vehicle_id
        self.interval_seconds = interval_ms / 1000
        self.state = VehicleState()
        self.running = False
        
    async def fetch_vehicle_id(self):
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.api_url}/api/telemetry/vehicles", timeout=5.0)
                if response.status_code != 200:
                    print(f"❌ Backend returned status {response.status_code}")
                    raise Exception(f"Backend error: {response.status_code}")
                
                vehicles = response.json()
                if vehicles:
                    self.vehicle_id = vehicles[0]["id"]
                    print(f"🚗 Vehicle: {vehicles[0]['year']} {vehicles[0]['make']} {vehicles[0]['model']} {vehicles[0].get('variant', '')}")
                    print(f"   VIN: {vehicles[0]['vin']}")
                else:
                    print("⚠️  No vehicles found. Creating default...")
                    await self.create_default_vehicle(client)
            except httpx.ConnectError:
                print("❌ Cannot connect to backend!")
                print("   Make sure: cd backend && python main.py")
                raise
            except Exception as e:
                print(f"❌ Error: {e}")
                raise
    
    async def create_default_vehicle(self, client: httpx.AsyncClient):
        print("   Waiting for database...")
        await asyncio.sleep(2)
        response = await client.get(f"{self.api_url}/api/telemetry/vehicles", timeout=5.0)
        vehicles = response.json()
        if vehicles:
            self.vehicle_id = vehicles[0]["id"]
            print(f"✅ Found vehicle: {vehicles[0]['make']} {vehicles[0]['model']}")
        else:
            raise Exception("No vehicles. Run: docker-compose down -v && docker-compose up -d")
    
    def start_engine(self):
        self.state.engine_running = True
        self.state.rpm = self.state.IDLE_RPM + random.randint(-50, 50)
        self.state.gear = 0
        self.state.battery_voltage = 14.2
        self.state.oil_pressure = 2.0
        # Reset ML metrics for new session
        self.state.ml = MLMetrics()
        print("🔑 Engine started")
    
    def select_gear(self, speed: float) -> int:
        if self.state.mode == DrivingMode.REVERSE:
            return -1
        if self.state.mode == DrivingMode.PARKED or speed < 1:
            return 0
        for gear in range(7, 0, -1):
            min_speed, max_speed = self.state.GEAR_SPEEDS[gear]
            if min_speed <= speed:
                return gear
        return 1
    
    def calculate_rpm(self, speed: float, gear: int) -> int:
        if gear <= 0 or speed < 1:
            return self.state.IDLE_RPM + random.randint(-30, 30)
        
        min_speed, max_speed = self.state.GEAR_SPEEDS[gear]
        speed_range = max_speed - min_speed
        speed_in_gear = max(0, speed - min_speed)
        rpm_range = self.state.MAX_RPM - self.state.IDLE_RPM
        
        rpm = self.state.IDLE_RPM + int((speed_in_gear / speed_range) * rpm_range)
        rpm = min(self.state.MAX_RPM, max(self.state.IDLE_RPM, rpm))
        rpm += random.randint(-50, 50)
        
        return rpm
    
    def set_mode(self, mode: DrivingMode):
        if mode == DrivingMode.REVERSE:
            if self.state.speed_kmh > 5:
                print("⚠️  Cannot reverse while moving!")
                return
        
        old_mode = self.state.mode
        self.state.mode = mode
        config = MODE_CONFIG[mode]
        
        if mode in [DrivingMode.CITY, DrivingMode.HIGHWAY, DrivingMode.SPORT]:
            self.state.target_speed = random.uniform(*config["target_speed_range"])
        elif mode == DrivingMode.PARKED:
            self.state.target_speed = 0
        elif mode == DrivingMode.REVERSE:
            self.state.target_speed = random.uniform(*config["target_speed_range"])
        
        if mode != old_mode:
            print(f"\n{config['color']}🔄 Mode: {config['name']}{' (Target: ' + str(int(self.state.target_speed)) + ' km/h)' if self.state.target_speed > 0 else ''}\033[0m\n")
    
    def update_speed(self, dt: float):
        if not self.state.engine_running:
            return
        
        config = MODE_CONFIG[self.state.mode]
        prev_speed_kmh = self.state.speed_kmh
        
        # Random target adjustments for realistic driving
        if self.state.mode in [DrivingMode.CITY, DrivingMode.HIGHWAY, DrivingMode.SPORT]:
            if random.random() < 0.05:
                self.state.target_speed = random.uniform(*config["target_speed_range"])
            
            # Random stops in city
            if self.state.mode == DrivingMode.CITY and random.random() < config["stop_probability"]:
                self.state.target_speed = 0
            elif self.state.mode == DrivingMode.CITY and self.state.target_speed == 0 and random.random() < 0.15:
                self.state.target_speed = random.uniform(*config["target_speed_range"])
        
        # Calculate speed change
        speed_diff = self.state.target_speed - self.state.speed_kmh
        
        if abs(speed_diff) < 0.5:
            self.state.speed_kmh = self.state.target_speed
            self.state.throttle_position = random.uniform(15, 30)
        elif speed_diff > 0:
            accel = config["acceleration"] * (1 + random.uniform(-0.2, 0.2))
            self.state.speed_kmh = min(self.state.target_speed, self.state.speed_kmh + accel * dt)
            self.state.throttle_position = min(100, 50 + (speed_diff / config["max_speed"]) * 100)
        else:
            decel = config["deceleration"] * (1 + random.uniform(-0.1, 0.1))
            self.state.speed_kmh = max(self.state.target_speed, self.state.speed_kmh - decel * dt)
            self.state.throttle_position = max(0, 10 - abs(speed_diff) * 0.5)
        
        # Ensure bounds
        self.state.speed_kmh = max(0, min(config["max_speed"], self.state.speed_kmh))
        
        # Update gear and RPM
        self.state.gear = self.select_gear(self.state.speed_kmh)
        self.state.rpm = self.calculate_rpm(self.state.speed_kmh, self.state.gear)
        
        # === ML METRICS UPDATE ===
        self.update_ml_metrics(prev_speed_kmh, dt)
    
    def update_ml_metrics(self, prev_speed_kmh: float, dt: float):
        """Calculate ML-ready derived metrics"""
        ml = self.state.ml
        
        # Convert speeds to m/s
        current_speed_ms = self.state.speed_kmh / 3.6
        prev_speed_ms = prev_speed_kmh / 3.6
        
        # Calculate acceleration (m/s²)
        if dt > 0:
            ml.acceleration_ms2 = (current_speed_ms - prev_speed_ms) / dt
        else:
            ml.acceleration_ms2 = 0.0
        
        # Convert to G-force (1g = 9.81 m/s²)
        ml.acceleration_g = ml.acceleration_ms2 / 9.81
        
        # Calculate jerk (rate of acceleration change)
        if dt > 0:
            ml.jerk_ms3 = (ml.acceleration_ms2 - ml._prev_acceleration) / dt
        ml._prev_acceleration = ml.acceleration_ms2
        
        # Update running averages
        ml._acceleration_samples += 1
        ml._acceleration_sum += abs(ml.acceleration_g)
        ml.avg_acceleration_g = ml._acceleration_sum / ml._acceleration_samples
        
        # Track max acceleration/deceleration
        if ml.acceleration_g > ml.max_acceleration_g:
            ml.max_acceleration_g = ml.acceleration_g
        if ml.acceleration_g < 0 and abs(ml.acceleration_g) > ml.max_deceleration_g:
            ml.max_deceleration_g = abs(ml.acceleration_g)
        
        # === EVENT DETECTION ===
        
        # Harsh braking detection (with edge detection to avoid counting same event multiple times)
        if ml.acceleration_g < ML_THRESHOLDS["harsh_brake_g"]:
            if not ml._in_harsh_brake:
                ml.harsh_brake_count += 1
                ml._in_harsh_brake = True
                print(f"\033[91m⚠️  HARSH BRAKE! ({ml.acceleration_g:.2f}g)\033[0m")
        else:
            ml._in_harsh_brake = False
        
        # Harsh acceleration detection
        if ml.acceleration_g > ML_THRESHOLDS["harsh_accel_g"]:
            if not ml._in_harsh_accel:
                ml.harsh_accel_count += 1
                ml._in_harsh_accel = True
                print(f"\033[93m⚡ HARSH ACCEL! ({ml.acceleration_g:.2f}g)\033[0m")
        else:
            ml._in_harsh_accel = False
        
        # Over-RPM tracking
        if self.state.rpm >= ML_THRESHOLDS["over_rpm"]:
            ml.over_rpm_seconds += int(dt)
        if self.state.rpm >= ML_THRESHOLDS["redline_rpm"]:
            ml.redline_seconds += int(dt)
        
        # Speeding tracking
        if self.state.speed_kmh > ML_THRESHOLDS["speeding_threshold"]:
            ml.over_speed_seconds += int(dt)
        
        # High throttle tracking
        if self.state.throttle_position >= ML_THRESHOLDS["high_throttle"]:
            ml.high_throttle_seconds += int(dt)
        
        # Idle vs moving time
        if self.state.speed_kmh < ML_THRESHOLDS["idle_speed"] and self.state.engine_running:
            ml.idle_seconds += int(dt)
        elif self.state.speed_kmh >= ML_THRESHOLDS["idle_speed"]:
            ml.moving_seconds += int(dt)
        
        # Calculate engine stress score (0-100)
        ml.engine_stress_score = self.calculate_engine_stress()
    
    def calculate_engine_stress(self) -> float:
        """Calculate a composite engine stress score (0-100)"""
        ml = self.state.ml
        
        # Factors contributing to engine stress
        rpm_factor = min(100, (self.state.rpm / self.state.MAX_RPM) * 100)
        temp_factor = min(100, max(0, (self.state.engine_temp - 80) / 40 * 100))
        throttle_factor = self.state.throttle_position
        
        # Weight the factors
        stress = (
            rpm_factor * 0.4 +
            temp_factor * 0.3 +
            throttle_factor * 0.2 +
            min(100, ml.max_acceleration_g * 100) * 0.1
        )
        
        return min(100, max(0, stress))
    
    def update_temperatures(self, dt: float):
        if self.state.speed_kmh > 0:
            target_engine_temp = 85 + (self.state.rpm / self.state.MAX_RPM) * 30
            if self.state.mode == DrivingMode.SPORT:
                target_engine_temp += 10
            target_oil_temp = target_engine_temp * 1.1
        else:
            target_engine_temp = 75 if self.state.engine_running else 20
            target_oil_temp = target_engine_temp * 1.05
        
        temp_change_rate = 0.3
        self.state.engine_temp += (target_engine_temp - self.state.engine_temp) * temp_change_rate * dt
        self.state.oil_temp += (target_oil_temp - self.state.oil_temp) * temp_change_rate * dt
        
        self.state.engine_temp += random.uniform(-0.5, 0.5)
        self.state.oil_temp += random.uniform(-0.3, 0.3)
        
        self.state.engine_temp = max(15, min(130, self.state.engine_temp))
        self.state.oil_temp = max(15, min(150, self.state.oil_temp))
    
    def update_oil_pressure(self):
        base_pressure = 1.5 + (self.state.rpm / 1000) * 0.4
        
        if self.state.oil_temp > 100:
            base_pressure *= 0.95
        elif self.state.oil_temp < 40:
            base_pressure *= 1.05
        
        base_pressure += random.uniform(-0.1, 0.1)
        self.state.oil_pressure = max(0.5, min(5.0, base_pressure))
    
    def update_fuel(self, dt: float):
        if self.state.speed_kmh > 0:
            base_consumption = 0.002
            speed_factor = self.state.speed_kmh / 100
            rpm_factor = self.state.rpm / 5000
            throttle_factor = self.state.throttle_position / 100
            
            consumption = base_consumption * (1 + speed_factor + rpm_factor * 0.5 + throttle_factor * 0.5) * dt
            self.state.fuel_level = max(0, self.state.fuel_level - consumption)
        else:
            # Idle consumption
            idle_consumption = 0.0005 * dt
            self.state.fuel_level = max(0, self.state.fuel_level - idle_consumption)
    
    def update_location(self, dt: float):
        if self.state.speed_kmh > 0:
            speed_ms = self.state.speed_kmh / 3.6
            distance = speed_ms * dt
            
            self.state.heading += random.uniform(-2, 2)
            self.state.heading = self.state.heading % 360
            
            heading_rad = math.radians(self.state.heading)
            delta_lat = (distance / 111000) * math.cos(heading_rad)
            delta_lon = (distance / (111000 * math.cos(math.radians(self.state.latitude)))) * math.sin(heading_rad)
            
            self.state.latitude += delta_lat
            self.state.longitude += delta_lon
    
    def update_tire_pressure(self, dt: float):
        speed_factor = self.state.speed_kmh / 200
        
        for attr in ['tire_pressure_fl', 'tire_pressure_fr', 'tire_pressure_rl', 'tire_pressure_rr']:
            base = getattr(self.state, attr)
            variation = random.uniform(-0.05, 0.05) + (speed_factor * 0.02)
            new_pressure = base + variation
            new_pressure = max(28.0, min(38.0, new_pressure))
            setattr(self.state, attr, round(new_pressure, 1))
    
    def tick(self, dt: float):
        if not self.state.engine_running:
            return
        
        self.update_speed(dt)
        self.update_temperatures(dt)
        self.update_oil_pressure()
        self.update_fuel(dt)
        self.update_location(dt)
        self.update_tire_pressure(dt)
    
    def get_telemetry(self) -> dict:
        """Get telemetry data with ML-ready metrics"""
        ml = self.state.ml
        
        return {
            "vehicle_id": self.vehicle_id,
            
            # Core telemetry
            "speed_kmh": round(self.state.speed_kmh, 1),
            "rpm": self.state.rpm,
            "gear": self.state.gear,
            "throttle_position": round(self.state.throttle_position, 1),
            "engine_temp": round(self.state.engine_temp, 1),
            "oil_temp": round(self.state.oil_temp, 1),
            "oil_pressure": round(self.state.oil_pressure, 2),
            "fuel_level": round(self.state.fuel_level, 1),
            "battery_voltage": round(self.state.battery_voltage, 1),
            
            # Tire pressure
            "tire_pressure_fl": round(self.state.tire_pressure_fl, 1),
            "tire_pressure_fr": round(self.state.tire_pressure_fr, 1),
            "tire_pressure_rl": round(self.state.tire_pressure_rl, 1),
            "tire_pressure_rr": round(self.state.tire_pressure_rr, 1),
            
            # Location
            "latitude": round(self.state.latitude, 6),
            "longitude": round(self.state.longitude, 6),
            "heading": round(self.state.heading, 1),
            "driving_mode": self.state.mode.value,
            
            # ML-ready derived metrics
            "acceleration_ms2": round(ml.acceleration_ms2, 3),
            "acceleration_g": round(ml.acceleration_g, 3),
            "jerk_ms3": round(ml.jerk_ms3, 3),
            "is_harsh_braking": ml._in_harsh_brake,
            "is_harsh_acceleration": ml._in_harsh_accel,
            "is_over_rpm": self.state.rpm >= ML_THRESHOLDS["over_rpm"],
            "is_speeding": self.state.speed_kmh > ML_THRESHOLDS["speeding_threshold"],
            "is_idling": self.state.speed_kmh < ML_THRESHOLDS["idle_speed"] and self.state.engine_running,
            "engine_stress_score": round(ml.engine_stress_score, 1),
        }
    
    def get_trip_ml_summary(self) -> dict:
        """Get ML metrics summary for trip end"""
        ml = self.state.ml
        total_time = ml.idle_seconds + ml.moving_seconds
        
        return {
            "harsh_brake_count": ml.harsh_brake_count,
            "harsh_accel_count": ml.harsh_accel_count,
            "over_rpm_seconds": ml.over_rpm_seconds,
            "redline_seconds": ml.redline_seconds,
            "over_speed_seconds": ml.over_speed_seconds,
            "high_throttle_seconds": ml.high_throttle_seconds,
            "idle_seconds": ml.idle_seconds,
            "moving_seconds": ml.moving_seconds,
            "idle_percentage": round((ml.idle_seconds / total_time * 100) if total_time > 0 else 0, 1),
            "avg_acceleration_g": round(ml.avg_acceleration_g, 3),
            "max_acceleration_g": round(ml.max_acceleration_g, 3),
            "max_deceleration_g": round(ml.max_deceleration_g, 3),
            "engine_stress_score": round(ml.engine_stress_score, 1),
        }
    
    async def send_telemetry(self):
        telemetry = self.get_telemetry()
        config = MODE_CONFIG[self.state.mode]
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/api/telemetry/reading",
                    json=telemetry
                )
                if response.status_code == 200:
                    gear_display = 'R' if telemetry['gear'] == -1 else ('N' if telemetry['gear'] == 0 else str(telemetry['gear']))
                    accel_str = f"{telemetry['acceleration_g']:+.2f}g" if abs(telemetry['acceleration_g']) > 0.01 else "     "
                    
                    # Show ML indicators
                    indicators = ""
                    if telemetry['is_harsh_braking']:
                        indicators += "🛑"
                    if telemetry['is_harsh_acceleration']:
                        indicators += "⚡"
                    if telemetry['is_over_rpm']:
                        indicators += "🔴"
                    if telemetry['is_speeding']:
                        indicators += "🚨"
                    
                    print(f"{config['color']}📡 [{config['name']:^7}] "
                          f"{telemetry['speed_kmh']:>5.1f} km/h | "
                          f"RPM: {telemetry['rpm']:>5} | "
                          f"G: {gear_display} | "
                          f"Accel: {accel_str} | "
                          f"🌡️ {telemetry['engine_temp']:.0f}°C | "
                          f"⛽ {telemetry['fuel_level']:.0f}% "
                          f"{indicators}\033[0m")
                else:
                    print(f"❌ Error: {response.status_code}")
            except httpx.ConnectError:
                print(f"❌ Cannot connect to backend")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def start_trip(self):
        if self.state.trip_active:
            return
            
        # Reset ML metrics for new trip
        self.state.ml = MLMetrics()
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/api/telemetry/trips/start",
                    json={
                        "vehicle_id": self.vehicle_id,
                        "start_latitude": self.state.latitude,
                        "start_longitude": self.state.longitude,
                    }
                )
                if response.status_code == 200:
                    trip = response.json()
                    self.state.trip_id = trip["id"]
                    self.state.trip_active = True
                    print(f"\n🚀 \033[92mTrip started!\033[0m (ID: {trip['id'][:8]}...)")
                    print(f"   ML metrics tracking: ✅ Acceleration, Harsh Events, Engine Stress\n")
            except Exception as e:
                pass
    
    async def end_trip(self):
        if not self.state.trip_active or not self.state.trip_id:
            return
        
        ml_summary = self.get_trip_ml_summary()
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/api/telemetry/trips/{self.state.trip_id}/end",
                    json={
                        "end_latitude": self.state.latitude,
                        "end_longitude": self.state.longitude,
                    }
                )
                if response.status_code == 200:
                    trip = response.json()
                    print(f"\n🏁 \033[92mTrip ended!\033[0m")
                    if trip.get('duration_seconds'):
                        print(f"   Duration: {trip['duration_seconds']}s")
                    if trip.get('distance_km'):
                        print(f"   Distance: {trip['distance_km']:.2f} km")
                    if trip.get('avg_speed_kmh'):
                        print(f"   Avg Speed: {trip['avg_speed_kmh']:.1f} km/h")
                    if trip.get('max_speed_kmh'):
                        print(f"   Max Speed: {trip['max_speed_kmh']:.1f} km/h")
                    
                    # Print ML Summary
                    print(f"\n   📊 \033[96mML Metrics Summary:\033[0m")
                    print(f"      Harsh Brakes: {ml_summary['harsh_brake_count']} | Harsh Accels: {ml_summary['harsh_accel_count']}")
                    print(f"      Max Accel: {ml_summary['max_acceleration_g']:.2f}g | Max Decel: {ml_summary['max_deceleration_g']:.2f}g")
                    print(f"      Over-RPM Time: {ml_summary['over_rpm_seconds']}s | Speeding Time: {ml_summary['over_speed_seconds']}s")
                    print(f"      Idle: {ml_summary['idle_percentage']:.1f}% | Engine Stress: {ml_summary['engine_stress_score']:.0f}/100")
                    print()
                    
                    self.state.trip_active = False
                    self.state.trip_id = None
            except Exception:
                pass
    
    def check_keyboard(self) -> str:
        if os.name == 'nt':
            if msvcrt.kbhit():
                try:
                    return msvcrt.getch().decode('utf-8', errors='ignore').lower()
                except:
                    return None
        return None
    
    async def run(self):
        print("\n" + "="*70)
        print("  🏎️  AutoPulse Porsche 911 Simulator v4 - ML Enhanced")
        print("="*70)
        
        await self.fetch_vehicle_id()
        
        print("\n📋 Controls:")
        print("   \033[93m[1] City Mode\033[0m    - Urban driving (25-50 km/h)")
        print("   \033[94m[2] Highway Mode\033[0m - Cruising (110-140 km/h)")
        print("   \033[91m[3] Sport Mode\033[0m   - Performance (140-220 km/h)")
        print("   \033[90m[P] Park\033[0m         - Stop the vehicle")
        print("   \033[95m[R] Reverse\033[0m      - Reverse (only when stopped)")
        print("   \033[33m[F] Low Fuel\033[0m     - Toggle low fuel warning")
        print("   \033[33m[T] High Temp\033[0m    - Toggle high engine temp")
        print("   [Q] Quit")
        print("\n📊 ML Metrics: Acceleration (g), Harsh Events, Engine Stress")
        print("-"*70)
        
        self.start_engine()
        await asyncio.sleep(1)
        
        self.state.mode = DrivingMode.PARKED
        self.state.target_speed = 0
        self.state.speed_kmh = 0
        self.running = True
        
        print(f"\n🅿️  Vehicle is PARKED - Press 1, 2, or 3 to start driving!")
        print(f"📡 Telemetry: 1 update/second with ML metrics")
        print("-"*70 + "\n")
        
        try:
            while self.running:
                key = self.check_keyboard()
                if key:
                    if key == '1':
                        self.set_mode(DrivingMode.CITY)
                    elif key == '2':
                        self.set_mode(DrivingMode.HIGHWAY)
                    elif key == '3':
                        self.set_mode(DrivingMode.SPORT)
                    elif key == 'p':
                        self.set_mode(DrivingMode.PARKED)
                    elif key == 'r':
                        self.set_mode(DrivingMode.REVERSE)
                    elif key == 'f':
                        if self.state.fuel_level > 20:
                            self.state.fuel_level = 8
                            print(f"\n\033[33m⚠️  LOW FUEL TEST: {self.state.fuel_level}%\033[0m\n")
                        else:
                            self.state.fuel_level = 85
                            print(f"\n\033[32m✅ FUEL RESET: {self.state.fuel_level}%\033[0m\n")
                    elif key == 't':
                        if self.state.engine_temp < 100:
                            self.state.engine_temp = 108
                            print(f"\n\033[33m⚠️  HIGH TEMP TEST: {self.state.engine_temp}°C\033[0m\n")
                        else:
                            self.state.engine_temp = 85
                            print(f"\n\033[32m✅ TEMP RESET: {self.state.engine_temp}°C\033[0m\n")
                    elif key == 'q':
                        print("\n🛑 Stopping simulator...")
                        self.running = False
                        break
                
                self.tick(self.interval_seconds)
                
                # Auto-start trip when moving forward
                if self.state.speed_kmh > 5 and self.state.mode not in [DrivingMode.PARKED, DrivingMode.REVERSE] and not self.state.trip_active:
                    await self.start_trip()
                
                await self.send_telemetry()
                await asyncio.sleep(self.interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted")
        finally:
            if self.state.trip_active:
                await self.end_trip()
            print("\n✅ Simulator stopped.\n")


async def main():
    simulator = Porsche911Simulator(
        api_url="http://localhost:8000",
        interval_ms=1000
    )
    await simulator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
