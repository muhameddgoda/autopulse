# AutoPulse - Quick Start Guide

## Prerequisites
- Docker Desktop installed and running
- Node.js 18+ installed
- Python 3.10+ installed

---

## Step 1: Start the Database

```bash
cd autopulse
docker-compose up -d
```

Wait a few seconds for the database to initialize.

Check if it's running:
```bash
docker ps
# Should see: autopulse_db running on port 5432
```

---

## Step 2: Start the Backend

```bash
cd autopulse/backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the backend
python main.py
```

The backend should start on `http://localhost:8000`

Test it:
- Open browser: http://localhost:8000
- Should see: `{"name": "AutoPulse", "version": "0.1.0", "status": "running"}`

---

## Step 3: Start the Vehicle Simulator

```bash
cd autopulse/simulator

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the simulator
python vehicle_simulator.py
```

Controls:
- `1` - City mode
- `2` - Highway mode
- `3` - Sport mode
- `P` - Park
- `R` - Reverse (when stopped)
- `Q` - Quit

---

## Step 4: Start the Frontend

```bash
cd autopulse/frontend

# Install dependencies (first time only)
npm install

# Run the frontend
npm run dev
```

Open browser: http://localhost:5173

---

## Step 5: Start the Driver HUD (Optional)

```bash
cd autopulse/driver-hud

# Install dependencies (first time only)
npm install

# Run the HUD
npm run dev
```

Open browser: http://localhost:5174

---

## Troubleshooting

### Backend won't start?
1. Check if Docker database is running: `docker ps`
2. Check port 8000 is free: `netstat -an | grep 8000`
3. Check Python dependencies: `pip install -r requirements.txt`

### Frontend shows "Offline"?
1. Make sure backend is running on port 8000
2. Check browser console for errors (F12)
3. Make sure simulator is running (it creates the vehicle)

### No data showing?
1. Start the simulator first - it creates the vehicle in the database
2. Switch between driving modes (1, 2, 3) to generate telemetry
3. Check WebSocket connection in browser console

### Database issues?
```bash
# Reset the database
docker-compose down -v
docker-compose up -d
```

### Trip not recording?
1. Make sure you started a trip on the Trips page
2. Or the simulator auto-starts trips when you switch modes

---

## Port Summary

| Service | Port |
|---------|------|
| Database (PostgreSQL) | 5432 |
| Backend API | 8000 |
| Frontend | 5173 |
| Driver HUD | 5174 |

---

## Quick Test Commands

```bash
# Test API health
curl http://localhost:8000/health

# Test vehicles endpoint
curl http://localhost:8000/api/telemetry/vehicles

# Test WebSocket (in browser console)
const ws = new WebSocket('ws://localhost:8000/api/telemetry/stream/YOUR_VEHICLE_ID');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```
