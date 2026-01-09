import { useState, useEffect } from "react";
import { useTelemetry } from "./hooks/useTelemetry";
import {
  Speedometer,
  RpmGauge,
  FuelGauge,
  GearDisplay,
  MapDisplay,
  AlertIcon,
  ConnectionStatus,
} from "./components/telemetry";
import { THEME, THRESHOLDS, TelemetryReading } from "./types";

// Drowsiness alert component
function DrowsinessAlert({ level }: { level: string }) {
  if (level === "none") return null;

  const isCritical = level === "critical" || level === "alert";
  const color = isCritical ? THEME.red : THEME.yellow;

  return (
    <div
      className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 px-8 py-4 rounded-xl font-bold text-2xl uppercase tracking-wider ${
        isCritical ? "animate-pulse" : ""
      }`}
      style={{
        backgroundColor: "rgba(0,0,0,0.9)",
        border: `3px solid ${color}`,
        color: color,
        boxShadow: `0 0 40px ${color}, inset 0 0 20px rgba(0,0,0,0.5)`,
      }}
    >
      <div className="flex items-center gap-3">
        <svg className="w-8 h-8" viewBox="0 0 24 24" fill={color}>
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
        </svg>
        <span>⚠️ DROWSINESS {level.toUpperCase()}</span>
      </div>
    </div>
  );
}

// API helper
async function fetchVehicleId(): Promise<string | null> {
  try {
    const res = await fetch("http://localhost:8000/api/telemetry/vehicles");
    if (!res.ok) throw new Error("Failed to fetch");
    const vehicles = await res.json();
    return vehicles[0]?.id || null;
  } catch (e) {
    console.error("Error fetching vehicle:", e);
    return null;
  }
}

// Default telemetry values
const DEFAULT_TELEMETRY: Partial<TelemetryReading> = {
  speed_kmh: 0,
  rpm: 800,
  gear: 0,
  fuel_level: 85,
  engine_temp: 20,
  oil_temp: 20,
  oil_pressure: 3.5,
  battery_voltage: 12.6,
  latitude: 48.8342,
  longitude: 9.1519,
  heading: 0,
  tire_pressure_fl: 33,
  tire_pressure_fr: 33,
  tire_pressure_rl: 32,
  tire_pressure_rr: 32,
};

export default function App() {
  const [vehicleId, setVehicleId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [drowsinessLevel, setDrowsinessLevel] = useState<string>("none");

  // Poll for drowsiness alerts
  useEffect(() => {
    const checkDrowsiness = async () => {
      try {
        // Check if there's an active safety session with drowsiness data
        const res = await fetch(
          `http://localhost:8000/api/safety/status/${
            vehicleId || "68f2ce4a-28df-4f11-bf5b-c961d1f7d064"
          }`
        );
        if (res.ok) {
          const data = await res.json();
          setDrowsinessLevel(data?.alert_level || "none");
        }
      } catch {
        // Silently fail - safety monitoring might not be active
      }
    };

    const interval = setInterval(checkDrowsiness, 500);
    return () => clearInterval(interval);
  }, [vehicleId]);

  // Fetch vehicle ID on mount
  useEffect(() => {
    const loadVehicle = async () => {
      const id = await fetchVehicleId();
      setVehicleId(id);
      setLoading(false);
    };
    loadVehicle();

    // Retry if no vehicle found
    const interval = setInterval(async () => {
      if (!vehicleId) {
        const id = await fetchVehicleId();
        if (id) setVehicleId(id);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [vehicleId]);

  const { telemetry, sensorHistory, connectionStatus } = useTelemetry({
    vehicleId,
    enabled: !!vehicleId,
  });

  // Merge telemetry with defaults
  const data = { ...DEFAULT_TELEMETRY, ...telemetry } as TelemetryReading;

  // Calculate alerts
  const alerts = {
    engine: data.engine_temp >= THRESHOLDS.engine_temp.warning,
    engineCritical: data.engine_temp >= THRESHOLDS.engine_temp.critical,
    oil: data.oil_temp >= THRESHOLDS.oil_temp.warning,
    oilCritical: data.oil_temp >= THRESHOLDS.oil_temp.critical,
    fuel: data.fuel_level <= THRESHOLDS.fuel_level.warning,
    fuelCritical: data.fuel_level <= THRESHOLDS.fuel_level.critical,
    battery: data.battery_voltage <= THRESHOLDS.battery_voltage.low,
    batteryCritical:
      data.battery_voltage <= THRESHOLDS.battery_voltage.critical,
    rpm: data.rpm >= THRESHOLDS.rpm.warning,
    rpmCritical: data.rpm >= THRESHOLDS.rpm.redline,
    tire:
      data.tire_pressure_fl <= THRESHOLDS.tire_pressure.low ||
      data.tire_pressure_fr <= THRESHOLDS.tire_pressure.low ||
      data.tire_pressure_rl <= THRESHOLDS.tire_pressure.low ||
      data.tire_pressure_rr <= THRESHOLDS.tire_pressure.low,
    tireCritical:
      data.tire_pressure_fl <= THRESHOLDS.tire_pressure.critical ||
      data.tire_pressure_fr <= THRESHOLDS.tire_pressure.critical ||
      data.tire_pressure_rl <= THRESHOLDS.tire_pressure.critical ||
      data.tire_pressure_rr <= THRESHOLDS.tire_pressure.critical,
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      {/* Drowsiness Warning Overlay */}
      <DrowsinessAlert level={drowsinessLevel} />

      {/* Main HUD Container */}
      <div
        className="rounded-2xl p-6 relative"
        style={{
          background: "linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%)",
          border: "1px solid #333",
          boxShadow: `0 0 60px ${THEME.orangeGlow}, inset 0 0 30px rgba(0,0,0,0.5)`,
        }}
      >
        {/* Top Bar - Alerts & Connection Status */}
        <div className="absolute top-4 left-4 right-4 flex justify-between items-center">
          {/* Left alerts */}
          <div className="flex gap-2">
            <AlertIcon
              type="fuel"
              active={alerts.fuel}
              critical={alerts.fuelCritical}
            />
            <AlertIcon
              type="engine"
              active={alerts.engine}
              critical={alerts.engineCritical}
            />
            <AlertIcon
              type="oil"
              active={alerts.oil}
              critical={alerts.oilCritical}
            />
          </div>

          {/* Right alerts + status */}
          <div className="flex gap-2 items-center">
            <AlertIcon
              type="battery"
              active={alerts.battery}
              critical={alerts.batteryCritical}
            />
            <AlertIcon
              type="tire"
              active={alerts.tire}
              critical={alerts.tireCritical}
            />
            <AlertIcon
              type="rpm"
              active={alerts.rpm}
              critical={alerts.rpmCritical}
            />
            <ConnectionStatus status={connectionStatus} />
          </div>
        </div>

        {/* Main Content */}
        <div className="flex gap-6 mt-8">
          {/* Center - Main Gauges */}
          <div className="flex items-end gap-4">
            <FuelGauge level={data.fuel_level} />
            <RpmGauge rpm={data.rpm} />
            <Speedometer speed={data.speed_kmh} />
            <MapDisplay
              lat={data.latitude ?? 48.8342}
              lng={data.longitude ?? 9.1519}
              heading={data.heading ?? 0}
            />
            <GearDisplay gear={data.gear} />
          </div>
        </div>

        {/* Bottom info bar (optional - timestamp) */}
        <div className="absolute bottom-3 left-0 right-10 flex justify-center">
          <div className="text-[10px] text-gray-600 font-mono">
            {telemetry?.time
              ? new Date(telemetry.time).toLocaleTimeString()
              : "--:--:--"}
          </div>
        </div>
      </div>
    </div>
  );
}
