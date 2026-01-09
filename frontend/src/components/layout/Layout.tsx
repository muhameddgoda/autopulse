import { useState, useEffect, useMemo } from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import AlertOverlay from "../AlertOverlay";
import { useTelemetry } from "../../hooks/useTelemetry";

interface Alert {
  type: "fuel" | "rpm" | "temperature" | "oil";
  message: string;
  severity: "warning" | "critical";
}

export default function Layout() {
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(
    new Set()
  );

  // Use the telemetry hook - it handles vehicle selection internally
  const [vehicleId, setVehicleId] = useState<string | null>(null);

  useEffect(() => {
    async function fetchVehicle() {
      try {
        const res = await fetch("http://localhost:8000/api/telemetry/vehicles");
        const vehicles = await res.json();
        if (vehicles.length > 0) setVehicleId(vehicles[0].id);
      } catch (e) {
        console.error("Failed to fetch vehicle:", e);
      }
    }
    fetchVehicle();
  }, []);

  const { telemetry } = useTelemetry({
    vehicleId,
    enabled: !!vehicleId,
  });

  // Generate alerts based on telemetry data
  const alerts = useMemo(() => {
    if (!telemetry) return [];

    const newAlerts: Alert[] = [];

    // Fuel warning
    if (telemetry.fuel_level < 15 && !dismissedAlerts.has("fuel")) {
      newAlerts.push({
        type: "fuel",
        message: `Low fuel: ${telemetry.fuel_level.toFixed(0)}%`,
        severity: telemetry.fuel_level < 10 ? "critical" : "warning",
      });
    }

    // RPM warning
    if (telemetry.rpm > 7000 && !dismissedAlerts.has("rpm")) {
      newAlerts.push({
        type: "rpm",
        message: `High RPM: ${telemetry.rpm.toFixed(0)}`,
        severity: telemetry.rpm > 7500 ? "critical" : "warning",
      });
    }

    // Temperature warning
    if (telemetry.engine_temp > 100 && !dismissedAlerts.has("temperature")) {
      newAlerts.push({
        type: "temperature",
        message: `High temperature: ${telemetry.engine_temp.toFixed(0)}°C`,
        severity: telemetry.engine_temp > 110 ? "critical" : "warning",
      });
    }

    // Oil pressure warning
    if (telemetry.oil_pressure < 20 && !dismissedAlerts.has("oil")) {
      newAlerts.push({
        type: "oil",
        message: `Low oil pressure: ${telemetry.oil_pressure.toFixed(0)} PSI`,
        severity: telemetry.oil_pressure < 15 ? "critical" : "warning",
      });
    }

    return newAlerts;
  }, [telemetry, dismissedAlerts]);

  // Reset dismissed alerts when telemetry values return to normal
  useEffect(() => {
    if (!telemetry) return;

    const newDismissed = new Set(dismissedAlerts);

    if (telemetry.fuel_level >= 15) newDismissed.delete("fuel");
    if (telemetry.rpm <= 7000) newDismissed.delete("rpm");
    if (telemetry.engine_temp <= 100) newDismissed.delete("temperature");
    if (telemetry.oil_pressure >= 20) newDismissed.delete("oil");

    if (newDismissed.size !== dismissedAlerts.size) {
      setDismissedAlerts(newDismissed);
    }
  }, [telemetry, dismissedAlerts]);

  // Dismiss an alert
  const handleDismiss = (index: number) => {
    const alert = alerts[index];
    if (alert) {
      setDismissedAlerts((prev) => new Set(prev).add(alert.type));
    }
  };

  return (
    <div
      className="h-screen flex overflow-hidden"
      style={{ backgroundColor: "#0a0a0a" }}
    >
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>

      {/* Alert Overlay */}
      <AlertOverlay alerts={alerts} onDismiss={handleDismiss} />
    </div>
  );
}
