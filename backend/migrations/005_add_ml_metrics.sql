-- Migration: Add ML metrics columns to telemetry_readings
-- Run this on existing database

ALTER TABLE telemetry_readings 
ADD COLUMN IF NOT EXISTS acceleration_ms2 FLOAT,
ADD COLUMN IF NOT EXISTS acceleration_g FLOAT,
ADD COLUMN IF NOT EXISTS jerk_ms3 FLOAT,
ADD COLUMN IF NOT EXISTS is_harsh_braking BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS is_harsh_acceleration BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS is_over_rpm BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS is_speeding BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS is_idling BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS engine_stress_score FLOAT;

-- Add ML metrics to trips table for trip summaries
ALTER TABLE trips
ADD COLUMN IF NOT EXISTS harsh_brake_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS harsh_accel_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS over_rpm_seconds INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS over_speed_seconds INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS idle_seconds INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS high_throttle_seconds INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS avg_acceleration_g FLOAT,
ADD COLUMN IF NOT EXISTS max_acceleration_g FLOAT,
ADD COLUMN IF NOT EXISTS max_deceleration_g FLOAT,
ADD COLUMN IF NOT EXISTS avg_engine_stress FLOAT,
ADD COLUMN IF NOT EXISTS driver_score FLOAT;

-- Create index for ML queries (finding harsh events)
CREATE INDEX IF NOT EXISTS idx_telemetry_harsh_events 
ON telemetry_readings (vehicle_id, time, is_harsh_braking, is_harsh_acceleration)
WHERE is_harsh_braking = TRUE OR is_harsh_acceleration = TRUE;

-- Create index for anomaly detection queries
CREATE INDEX IF NOT EXISTS idx_telemetry_anomaly 
ON telemetry_readings (vehicle_id, time, engine_stress_score)
WHERE engine_stress_score > 70;
