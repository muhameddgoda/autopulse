-- Migration: Enhanced Trip Tracking
-- Adds mode breakdown and more stats to trips table

-- Add new columns to trips table
ALTER TABLE trips ADD COLUMN IF NOT EXISTS avg_rpm INTEGER;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS fuel_start FLOAT;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS fuel_end FLOAT;

-- Mode breakdown columns (seconds spent in each mode)
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_parked_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_city_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_highway_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_sport_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_reverse_seconds INTEGER DEFAULT 0;

-- Running totals for calculating averages
ALTER TABLE trips ADD COLUMN IF NOT EXISTS total_readings INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS speed_sum FLOAT DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS rpm_sum FLOAT DEFAULT 0;

-- Set defaults for existing rows
UPDATE trips SET 
    mode_parked_seconds = COALESCE(mode_parked_seconds, 0),
    mode_city_seconds = COALESCE(mode_city_seconds, 0),
    mode_highway_seconds = COALESCE(mode_highway_seconds, 0),
    mode_sport_seconds = COALESCE(mode_sport_seconds, 0),
    mode_reverse_seconds = COALESCE(mode_reverse_seconds, 0),
    total_readings = COALESCE(total_readings, 0),
    speed_sum = COALESCE(speed_sum, 0),
    rpm_sum = COALESCE(rpm_sum, 0);
