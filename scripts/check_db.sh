#!/bin/bash
# AutoPulse Database Check and Fix Script
# Run this in Git Bash or PowerShell

echo "========================================"
echo "  AutoPulse Database Check & Fix"
echo "========================================"
echo ""

# Check if docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if autopulse_db container exists
if ! docker ps -a | grep -q autopulse_db; then
    echo "❌ autopulse_db container not found."
    echo "   Run: docker-compose up -d"
    exit 1
fi

echo "✅ Docker is running"
echo "✅ autopulse_db container found"
echo ""

# Run migrations
echo "Running database migrations..."
echo ""

docker exec autopulse_db psql -U autopulse -d autopulse -c "
-- Check existing columns
SELECT column_name FROM information_schema.columns WHERE table_name = 'trips';
"

echo ""
echo "Adding missing columns..."

docker exec autopulse_db psql -U autopulse -d autopulse -c "
-- Add missing columns to trips table
ALTER TABLE trips ADD COLUMN IF NOT EXISTS avg_rpm INTEGER;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS fuel_start FLOAT;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS fuel_end FLOAT;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_parked_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_city_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_highway_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_sport_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS mode_reverse_seconds INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS total_readings INTEGER DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS speed_sum FLOAT DEFAULT 0;
ALTER TABLE trips ADD COLUMN IF NOT EXISTS rpm_sum FLOAT DEFAULT 0;

-- Add driving_mode to telemetry_readings if missing
ALTER TABLE telemetry_readings ADD COLUMN IF NOT EXISTS driving_mode VARCHAR(20);
"

echo ""
echo "✅ Migrations complete!"
echo ""

# Check trip data
echo "Checking trip data..."
docker exec autopulse_db psql -U autopulse -d autopulse -c "
SELECT id, is_active, duration_seconds, distance_km, mode_city_seconds, mode_sport_seconds 
FROM trips 
ORDER BY created_at DESC 
LIMIT 5;
"

echo ""
echo "Checking telemetry readings..."
docker exec autopulse_db psql -U autopulse -d autopulse -c "
SELECT COUNT(*) as total_readings, 
       COUNT(DISTINCT driving_mode) as unique_modes,
       MIN(time) as oldest,
       MAX(time) as newest
FROM telemetry_readings;
"

echo ""
echo "========================================"
echo "  Done! Restart your backend:"
echo "  cd backend && uvicorn main:app --reload"
echo "========================================"
