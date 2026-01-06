#!/bin/bash
# AutoPulse Status Checker & Troubleshooter

echo "🚗 AutoPulse Status Check"
echo "========================="
echo ""

# Check Docker
echo "📦 Docker Status:"
if docker ps | grep -q autopulse_db; then
    echo "   ✅ Database container running"
else
    echo "   ❌ Database NOT running"
    echo "   → Run: docker-compose up -d"
fi
echo ""

# Check Backend
echo "🔧 Backend Status:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ Backend running on port 8000"
    # Check vehicles
    VEHICLES=$(curl -s http://localhost:8000/api/telemetry/vehicles)
    if [ "$VEHICLES" = "[]" ]; then
        echo "   ⚠️  No vehicles in database (start simulator first)"
    else
        echo "   ✅ Vehicles found in database"
    fi
else
    echo "   ❌ Backend NOT running"
    echo "   → Run: cd backend && python main.py"
fi
echo ""

# Check Frontend
echo "🖥️  Frontend Status:"
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "   ✅ Frontend running on port 5173"
else
    echo "   ❌ Frontend NOT running"
    echo "   → Run: cd frontend && npm run dev"
fi
echo ""

# Check Driver HUD
echo "🎮 Driver HUD Status:"
if curl -s http://localhost:5174 > /dev/null 2>&1; then
    echo "   ✅ Driver HUD running on port 5174"
else
    echo "   ⚪ Driver HUD not running (optional)"
    echo "   → Run: cd driver-hud && npm run dev"
fi
echo ""

# Summary
echo "========================="
echo "📋 Quick Fix Commands:"
echo ""
echo "1. Start everything:"
echo "   docker-compose up -d"
echo "   cd backend && python main.py &"
echo "   cd simulator && python vehicle_simulator.py &"
echo "   cd frontend && npm run dev &"
echo ""
echo "2. Reset database:"
echo "   docker-compose down -v && docker-compose up -d"
echo ""
echo "3. Check logs:"
echo "   docker logs autopulse_db"
