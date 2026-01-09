-- Migration: Add driver scoring columns to trips table
-- Run this on your PostgreSQL database

-- Add driver behavior scoring columns
ALTER TABLE trips 
ADD COLUMN IF NOT EXISTS driver_score FLOAT,
ADD COLUMN IF NOT EXISTS behavior_label VARCHAR(20),
ADD COLUMN IF NOT EXISTS risk_level VARCHAR(20);

-- Add harsh event counters (for quick access without recalculating)
ALTER TABLE trips
ADD COLUMN IF NOT EXISTS harsh_brake_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS harsh_accel_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS speeding_percentage FLOAT DEFAULT 0;

-- Add index for efficient score-based queries
CREATE INDEX IF NOT EXISTS idx_trips_driver_score ON trips (driver_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_trips_behavior ON trips (behavior_label);

-- Update existing trips with default values (optional - backfill)
-- UPDATE trips SET driver_score = NULL, behavior_label = NULL, risk_level = NULL WHERE driver_score IS NULL;

COMMENT ON COLUMN trips.driver_score IS 'Overall driver behavior score 0-100, calculated at trip end';
COMMENT ON COLUMN trips.behavior_label IS 'Behavior classification: exemplary, calm, normal, aggressive, dangerous';
COMMENT ON COLUMN trips.risk_level IS 'Risk assessment: low, medium, high, critical';
