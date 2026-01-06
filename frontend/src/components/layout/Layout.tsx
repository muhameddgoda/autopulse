import { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import { AlertOverlay, useAlerts } from '../AlertOverlay';
import { useTelemetry } from '../../hooks/useTelemetry';
import { vehicleApi } from '../../lib/api';
import { Vehicle } from '../../types';

export default function Layout() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);

  useEffect(() => {
    async function fetchVehicle() {
      try {
        const vehicles = await vehicleApi.getAll();
        if (vehicles.length > 0) setVehicle(vehicles[0]);
      } catch (error) {
        console.error('Failed to fetch vehicle:', error);
      }
    }
    fetchVehicle();
  }, []);

  const { telemetry } = useTelemetry({
    vehicleId: vehicle?.id ?? null,
    enabled: !!vehicle,
  });

  const { currentAlert, dismissAlert } = useAlerts(telemetry);

  return (
    <div className="h-screen flex overflow-hidden" style={{ backgroundColor: '#0a0a0a' }}>
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>

      {/* Alert Overlay */}
      <AlertOverlay alert={currentAlert} onDismiss={dismissAlert} />
    </div>
  );
}
