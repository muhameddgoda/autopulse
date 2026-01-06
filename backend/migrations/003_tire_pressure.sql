-- Migration: Add tire pressure and heading to telemetry
-- Run this after the database is created

-- Add tire pressure columns to telemetry_readings
ALTER TABLE telemetry_readings 
ADD COLUMN IF NOT EXISTS tire_pressure_fl FLOAT DEFAULT 33.0,
ADD COLUMN IF NOT EXISTS tire_pressure_fr FLOAT DEFAULT 33.0,
ADD COLUMN IF NOT EXISTS tire_pressure_rl FLOAT DEFAULT 32.0,
ADD COLUMN IF NOT EXISTS tire_pressure_rr FLOAT DEFAULT 32.0,
ADD COLUMN IF NOT EXISTS heading FLOAT DEFAULT 0;

-- Verify columns were added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'telemetry_readings' 
AND column_name IN ('tire_pressure_fl', 'tire_pressure_fr', 'tire_pressure_rl', 'tire_pressure_rr', 'heading');
