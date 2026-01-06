# 🏎️ AutoPulse - Connected Car Platform

<div align="center">

![AutoPulse Banner](docs/images/banner.png)

**Real-time vehicle telemetry platform for Porsche 911**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6.svg)](https://typescriptlang.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)](https://postgresql.org)

[Features](#features) • [Demo](#demo) • [Architecture](#architecture) • [Setup](#quick-start) • [Modules](#modules)

</div>

---

## 🎯 Overview

AutoPulse is a **connected car platform** that demonstrates real-time vehicle telemetry processing, visualization, and analytics. Built as a portfolio project showcasing skills in:

- **Backend Development**: FastAPI, WebSockets, PostgreSQL
- **Frontend Development**: React, TypeScript, Three.js, Recharts
- **Real-time Systems**: WebSocket streaming, live data visualization
- **Data Engineering**: Time-series data, analytics pipelines
- **3D Visualization**: Interactive 3D car model with dynamic lighting

---

## ✨ Features

### 📊 Real-Time Dashboard
- Live telemetry streaming via WebSocket
- Mode-themed UI (City/Highway/Sport)
- Interactive 3D Porsche 911 model
- Dynamic warning system (Fuel, RPM, Oil, Temperature)

### 🗺️ Live Map Tracking
- Real-time GPS position updates
- Speed overlay on map
- Route visualization

### 📈 Analytics & Charts
- 60-second rolling telemetry history
- Speed, RPM, Temperature, Throttle charts
- Trip analytics with mode breakdown

### 🚗 Trip Management
- Auto-trip recording when driving
- Mode breakdown (time in City/Highway/Sport)
- Weekly statistics dashboard
- CSV export for ML training

### 🔔 Smart Alerts
- Low fuel warning (< 15%)
- RPM redline warning (> 7500)
- Engine temperature alerts
- Oil pressure monitoring

---

## 🎬 Demo

### Main Dashboard - Sport Mode
![Sport Mode](docs/images/sport-mode.png)

### Trip Analytics
![Trip Analytics](docs/images/trips.png)

### Live Charts
![Live Charts](docs/images/charts.png)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │Dashboard │ │   Map    │ │  Charts  │ │  Trips   │          │
│  │ + 3D Car │ │  (Leaflet)│ │(Recharts)│ │Analytics │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       └────────────┴────────────┴────────────┘                 │
│                           │ WebSocket                          │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                    Backend (FastAPI)                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   REST API + WebSocket                    │  │
│  │  /api/telemetry/reading    - Store telemetry             │  │
│  │  /api/telemetry/stream     - WebSocket streaming         │  │
│  │  /api/telemetry/trips      - Trip management             │  │
│  │  /api/telemetry/export/csv - ML data export              │  │
│  └────────────────────────────┬─────────────────────────────┘  │
│                               │                                 │
│  ┌────────────────────────────┴─────────────────────────────┐  │
│  │                 PostgreSQL Database                       │  │
│  │  vehicles │ telemetry_readings │ trips                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ HTTP POST
┌───────────────────────────┼─────────────────────────────────────┐
│                Vehicle Simulator (Python)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Porsche 911 Physics Engine                              │  │
│  │  - Keyboard Control (1/2/3 = City/Highway/Sport)         │  │
│  │  - Realistic acceleration/deceleration                   │  │
│  │  - Auto trip recording                                   │  │
│  │  - 1 Hz telemetry updates                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/autopulse.git
cd autopulse

# Start database
docker-compose up -d

# Run migrations
./scripts/check_db.sh
```

### 2. Start Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Start Driver HUD (Optional)

```bash
cd driver-hud
npm install
npm run dev
```

### 5. Run Simulator

```bash
cd simulator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python vehicle_simulator.py
```

### 6. Open Dashboards

- **Main Dashboard**: http://localhost:5173
- **Driver HUD**: http://localhost:5174
- **API Docs**: http://localhost:8000/docs

---

## 🎮 Simulator Controls

| Key | Action |
|-----|--------|
| `1` | City Mode (25-50 km/h) |
| `2` | Highway Mode (110-140 km/h) |
| `3` | Sport Mode (140-220 km/h) |
| `P` | Park |
| `R` | Reverse (only when stopped) |
| `F` | Toggle low fuel (test warnings) |
| `Q` | Quit |

---

## 📦 Modules

### Module 1: Telemetry Platform ✅
Real-time data collection, visualization, and analytics.

### Module 2: Predictive Maintenance 🚧
ML-based maintenance prediction using telemetry patterns.

### Module 3: Drowsiness Detection 🚧
Computer vision for driver alertness monitoring.

### Module 4: Driver Behavior Analytics 🚧
Causal analysis of driving patterns for maintenance forecasting.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, TypeScript, Tailwind CSS, Three.js, Recharts |
| Backend | FastAPI, SQLAlchemy, WebSockets |
| Database | PostgreSQL 15 |
| 3D Model | GLTF/GLB, React Three Fiber |
| Maps | Leaflet, OpenStreetMap |
| Simulator | Python, asyncio, httpx |

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/telemetry/vehicles` | List all vehicles |
| POST | `/api/telemetry/reading` | Submit telemetry reading |
| GET | `/api/telemetry/latest/{vehicle_id}` | Get latest reading |
| WS | `/api/telemetry/stream/{vehicle_id}` | WebSocket stream |
| POST | `/api/telemetry/trips/start` | Start new trip |
| POST | `/api/telemetry/trips/{trip_id}/end` | End trip |
| GET | `/api/telemetry/stats/weekly/{vehicle_id}` | Weekly stats |
| GET | `/api/telemetry/export/csv/{vehicle_id}` | Export telemetry CSV |
| GET | `/api/telemetry/export/trips-csv/{vehicle_id}` | Export trips CSV |

---

## 📁 Project Structure

```
autopulse/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes
│   │   ├── models/         # SQLAlchemy models
│   │   └── schemas/        # Pydantic schemas
│   ├── migrations/         # SQL migrations
│   └── main.py            # App entry point
├── frontend/
│   ├── src/
│   │   ├── pages/         # React pages
│   │   ├── components/    # Reusable components
│   │   ├── hooks/         # Custom hooks
│   │   └── types/         # TypeScript types
│   └── public/models/     # 3D car model
├── driver-hud/            # Standalone HUD app
├── simulator/             # Vehicle simulator
├── scripts/               # Utility scripts
└── docker-compose.yml
```

---

## 🎨 Mode Themes

| Mode | Color | Speed Range |
|------|-------|-------------|
| 🅿️ Parked | Gray | 0 km/h |
| 🔄 Reverse | Purple | 0-15 km/h |
| 🏙️ City | Cyan | 25-50 km/h |
| 🛣️ Highway | Blue | 110-140 km/h |
| 🔥 Sport | Orange | 140-220 km/h |

---

## 👤 Author

**Mohamed** - BSc Robotics and Intelligent Systems, Constructor University Bremen

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the automotive industry**

🏎️ *Ready to drive innovation at Porsche, BMW, or Mercedes?* 🏎️

</div>
