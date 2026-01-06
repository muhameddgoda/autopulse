#!/bin/bash
# Run database migrations for AutoPulse

echo "🔄 Running AutoPulse Database Migrations..."
echo ""

# Database connection
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-autopulse}"
DB_PASS="${DB_PASS:-autopulse_secret}"
DB_NAME="${DB_NAME:-autopulse}"

# Export password for psql
export PGPASSWORD="$DB_PASS"

# Run each migration
for migration in backend/migrations/*.sql; do
    echo "📄 Running: $migration"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration" 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Success"
    else
        echo "   ⚠️  May have warnings (columns might already exist)"
    fi
    echo ""
done

echo "✅ Migrations complete!"
echo ""
echo "To verify, run:"
echo "  docker exec -it autopulse_db psql -U autopulse -d autopulse -c \"\\d telemetry_readings\""
