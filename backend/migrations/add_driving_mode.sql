-- Add driving_mode column to telemetry_readings
-- Run this in your PostgreSQL database

ALTER TABLE telemetry_readings 
ADD COLUMN IF NOT EXISTS driving_mode VARCHAR(20);
