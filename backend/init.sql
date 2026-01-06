-- AutoPulse Database Schema
-- PostgreSQL + TimescaleDB for Porsche 911 Telemetry

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- VEHICLES TABLE
-- ============================================
CREATE TABLE vehicles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vin VARCHAR(17) UNIQUE NOT NULL,
    make VARCHAR(50) DEFAULT 'Porsche',
    model VARCHAR(50) DEFAULT '911',
    variant VARCHAR(50),  -- e.g., 'Carrera', 'Turbo S', 'GT3'
    year INTEGER,
    color VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- TELEMETRY READINGS (Time-Series Hypertable)
-- ============================================
CREATE TABLE telemetry_readings (
    time TIMESTAMPTZ NOT NULL,
    vehicle_id UUID NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
    
    -- Core driving data
    speed_kmh FLOAT,
    rpm INTEGER,
    gear SMALLINT,  -- -1=R, 0=N, 1-7 for PDK
    throttle_position FLOAT,
    
    -- Engine health
    engine_temp FLOAT,
    oil_temp FLOAT,
    oil_pressure FLOAT,
    
    -- Fuel & electrical
    fuel_level FLOAT,
    battery_voltage FLOAT,
    
    -- Tire pressure (PSI)
    tire_pressure_fl FLOAT DEFAULT 33.0,
    tire_pressure_fr FLOAT DEFAULT 33.0,
    tire_pressure_rl FLOAT DEFAULT 32.0,
    tire_pressure_rr FLOAT DEFAULT 32.0,
    
    -- Location
    latitude FLOAT,
    longitude FLOAT,
    heading FLOAT DEFAULT 0,
    
    -- Driving mode
    driving_mode VARCHAR(20),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable for efficient time-series queries
SELECT create_hypertable('telemetry_readings', 'time');

-- Create index for fast vehicle lookups
CREATE INDEX idx_telemetry_vehicle_time ON telemetry_readings (vehicle_id, time DESC);

-- ============================================
-- TRIPS TABLE
-- ============================================
CREATE TABLE trips (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vehicle_id UUID NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
    
    -- Trip timing
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Trip statistics (calculated on trip end)
    distance_km FLOAT,
    duration_seconds INTEGER,
    avg_speed_kmh FLOAT,
    max_speed_kmh FLOAT,
    max_rpm INTEGER,
    fuel_used_liters FLOAT,
    
    -- Route (start and end points)
    start_latitude FLOAT,
    start_longitude FLOAT,
    end_latitude FLOAT,
    end_longitude FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for active trip lookups
CREATE INDEX idx_trips_active ON trips (vehicle_id, is_active) WHERE is_active = TRUE;

-- ============================================
-- INSERT DEFAULT VEHICLE (Porsche 911 Carrera)
-- ============================================
INSERT INTO vehicles (vin, make, model, variant, year, color)
VALUES ('WP0AB2A91NS123456', 'Porsche', '911', 'Turbo S', 2024, 'Signal Orange');

-- ============================================
-- USEFUL VIEWS
-- ============================================

-- Latest reading per vehicle
CREATE VIEW latest_telemetry AS
SELECT DISTINCT ON (vehicle_id) *
FROM telemetry_readings
ORDER BY vehicle_id, time DESC;

-- Active trips
CREATE VIEW active_trips AS
SELECT * FROM trips WHERE is_active = TRUE;
