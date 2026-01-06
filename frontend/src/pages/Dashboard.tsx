import { useState, useEffect } from 'react';
import { 
  Gauge, 
  Thermometer, 
  Droplets, 
  Battery, 
  Fuel, 
  Activity,
  Wifi,
  WifiOff,
  Car,
  Navigation
} from 'lucide-react';
import { useTelemetry } from '../hooks/useTelemetry';
import { vehicleApi } from '../lib/api';
import { Vehicle, getGearDisplay, THRESHOLDS, getStatus } from '../types';
import { 
  cn, 
  formatSpeed, 
  formatRPM, 
  formatTemp, 
  formatPressure, 
  formatVoltage, 
  formatPercent,
  getStatusColor,
  calculatePercentage
} from '../lib/utils';
import LiveMap from '../components/LiveMap';

// Compact Gauge Component
interface GaugeDisplayProps {
  label: string;
  value: string;
  unit?: string;
  icon: React.ReactNode;
  percentage?: number;
  status?: 'normal' | 'warning' | 'critical';
}

function GaugeDisplay({ label, value, unit, icon, percentage, status = 'normal' }: GaugeDisplayProps) {
  return (
    <div className="bg-porsche-gray-800 rounded-xl border border-porsche-gray-700 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-porsche-gray-400 text-xs font-medium">{label}</span>
        <span className={cn('opacity-60', getStatusColor(status))}>{icon}</span>
      </div>
      
      <div className="flex items-baseline gap-1">
        <span className={cn('text-2xl font-bold font-mono', getStatusColor(status))}>
          {value}
        </span>
        {unit && <span className="text-porsche-gray-500 text-xs">{unit}</span>}
      </div>

      {percentage !== undefined && (
        <div className="mt-2 h-1 bg-porsche-gray-700 rounded-full overflow-hidden">
          <div 
            className={cn(
              'h-full rounded-full transition-all duration-300',
              status === 'critical' ? 'bg-red-500' : 
              status === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
            )}
            style={{ width: `${Math.min(100, percentage)}%` }}
          />
        </div>
      )}
    </div>
  );
}

// Large Speed Display Overlay
interface SpeedOverlayProps {
  speed: number;
  rpm: number;
  gear: number;
}

function SpeedOverlay({ speed, rpm, gear }: SpeedOverlayProps) {
  const isRedline = rpm >= 7500;
  
  return (
    <div className="absolute top-4 left-4 z-[1000] flex flex-col gap-2">
      {/* Speed */}
      <div className="bg-black/80 backdrop-blur-sm rounded-xl px-6 py-4">
        <div className="text-porsche-gray-400 text-xs mb-1">SPEED</div>
        <div className="flex items-baseline">
          <span className="text-6xl font-bold font-mono text-white">{formatSpeed(speed)}</span>
          <span className="text-porsche-gray-400 text-lg ml-2">km/h</span>
        </div>
      </div>
      
      {/* RPM */}
      <div className="bg-black/80 backdrop-blur-sm rounded-xl px-6 py-3">
        <div className="text-porsche-gray-400 text-xs mb-1">RPM</div>
        <div className={cn(
          'text-4xl font-bold font-mono',
          isRedline ? 'text-red-500' : 'text-white'
        )}>
          {formatRPM(rpm)}
        </div>
        {/* RPM Bar */}
        <div className="mt-2 h-1.5 bg-porsche-gray-700 rounded-full overflow-hidden w-48">
          <div 
            className={cn(
              'h-full rounded-full transition-all duration-100',
              isRedline ? 'bg-red-500' : 'bg-porsche-red'
            )}
            style={{ width: `${(rpm / 8500) * 100}%` }}
          />
        </div>
      </div>
      
      {/* Gear */}
      <div className="bg-black/80 backdrop-blur-sm rounded-xl px-6 py-3">
        <div className="text-porsche-gray-400 text-xs mb-1">GEAR</div>
        <div className={cn(
          'text-5xl font-bold font-mono',
          gear === -1 ? 'text-yellow-500' : 'text-white'
        )}>
          {getGearDisplay(gear)}
        </div>
      </div>
    </div>
  );
}

// Connection Status Badge
interface ConnectionStatusProps {
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
}

function ConnectionStatus({ status }: ConnectionStatusProps) {
  const statusConfig = {
    connecting: { color: 'text-yellow-500', bg: 'bg-yellow-500/20', icon: <Wifi className="w-4 h-4 animate-pulse" />, text: 'Connecting...' },
    connected: { color: 'text-green-500', bg: 'bg-green-500/20', icon: <Wifi className="w-4 h-4" />, text: 'Live' },
    disconnected: { color: 'text-gray-500', bg: 'bg-gray-500/20', icon: <WifiOff className="w-4 h-4" />, text: 'Disconnected' },
    error: { color: 'text-red-500', bg: 'bg-red-500/20', icon: <WifiOff className="w-4 h-4" />, text: 'Error' },
  };

  const config = statusConfig[status];

  return (
    <div className={cn('flex items-center gap-2 px-3 py-1.5 rounded-full', config.bg, config.color)}>
      {config.icon}
      <span className="text-sm font-medium">{config.text}</span>
    </div>
  );
}

// Main Dashboard Component
export default function Dashboard() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch vehicle on mount
  useEffect(() => {
    async function fetchVehicle() {
      try {
        const vehicles = await vehicleApi.getAll();
        if (vehicles.length > 0) {
          setVehicle(vehicles[0]);
        }
      } catch (error) {
        console.error('Failed to fetch vehicle:', error);
      } finally {
        setLoading(false);
      }
    }
    fetchVehicle();
  }, []);

  // Connect to WebSocket for real-time telemetry
  const { telemetry, connectionStatus } = useTelemetry({
    vehicleId: vehicle?.id ?? null,
    enabled: !!vehicle,
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-porsche-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!vehicle) {
    return (
      <div className="min-h-screen bg-porsche-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">No vehicle found</div>
      </div>
    );
  }

  // Use telemetry data or defaults
  const data = telemetry ?? {
    speed_kmh: 0,
    rpm: 0,
    gear: 0,
    throttle_position: 0,
    engine_temp: 20,
    oil_temp: 20,
    oil_pressure: 0,
    fuel_level: 0,
    battery_voltage: 12.6,
    latitude: 48.8342,
    longitude: 9.1519,
  };

  return (
    <div className="h-screen bg-porsche-gray-900 text-white p-4 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between mb-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-porsche-red rounded-lg flex items-center justify-center">
            <Car className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold">AutoPulse</h1>
            <p className="text-porsche-gray-400 text-xs">
              {vehicle.year} {vehicle.make} {vehicle.model} {vehicle.variant}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-porsche-gray-400 text-sm">
            <Navigation className="w-4 h-4" />
            <span>Stuttgart, Germany</span>
          </div>
          <ConnectionStatus status={connectionStatus} />
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
        {/* Left: Live Map with Overlays */}
        <div className="col-span-8 relative rounded-xl overflow-hidden">
          <LiveMap 
            latitude={data.latitude ?? 48.8342}
            longitude={data.longitude ?? 9.1519}
            speed={data.speed_kmh}
          />
          
          {/* Speed/RPM/Gear Overlay on Map */}
          <SpeedOverlay 
            speed={data.speed_kmh} 
            rpm={data.rpm} 
            gear={data.gear} 
          />
        </div>

        {/* Right: Gauges Panel */}
        <div className="col-span-4 flex flex-col gap-3">
          {/* Fuel - Large */}
          <div className="bg-porsche-gray-800 rounded-xl border border-porsche-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-porsche-gray-400 text-sm font-medium">FUEL LEVEL</span>
              <Fuel className={cn('w-5 h-5', getStatusColor(getStatus(data.fuel_level, THRESHOLDS.fuel_level)))} />
            </div>
            <div className="text-4xl font-bold font-mono mb-2">
              {formatPercent(data.fuel_level)}
            </div>
            <div className="h-2 bg-porsche-gray-700 rounded-full overflow-hidden">
              <div 
                className={cn(
                  'h-full rounded-full transition-all',
                  data.fuel_level < 15 ? 'bg-red-500' : 
                  data.fuel_level < 30 ? 'bg-yellow-500' : 'bg-green-500'
                )}
                style={{ width: `${data.fuel_level}%` }}
              />
            </div>
          </div>

          {/* Engine Stats Grid */}
          <div className="grid grid-cols-2 gap-3">
            <GaugeDisplay
              label="ENGINE"
              value={formatTemp(data.engine_temp)}
              icon={<Thermometer className="w-4 h-4" />}
              percentage={calculatePercentage(data.engine_temp, 0, 130)}
              status={getStatus(data.engine_temp, THRESHOLDS.engine_temp)}
            />
            <GaugeDisplay
              label="OIL TEMP"
              value={formatTemp(data.oil_temp)}
              icon={<Droplets className="w-4 h-4" />}
              percentage={calculatePercentage(data.oil_temp, 0, 150)}
              status={getStatus(data.oil_temp, THRESHOLDS.oil_temp)}
            />
            <GaugeDisplay
              label="OIL PRESS"
              value={formatPressure(data.oil_pressure)}
              icon={<Gauge className="w-4 h-4" />}
              percentage={calculatePercentage(data.oil_pressure, 0, 6)}
              status={getStatus(data.oil_pressure, THRESHOLDS.oil_pressure)}
            />
            <GaugeDisplay
              label="THROTTLE"
              value={formatPercent(data.throttle_position)}
              icon={<Activity className="w-4 h-4" />}
              percentage={data.throttle_position}
            />
          </div>

          {/* Battery */}
          <div className="bg-porsche-gray-800 rounded-xl border border-porsche-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-porsche-gray-400 text-sm font-medium">BATTERY</span>
              <Battery className={cn('w-5 h-5', getStatusColor(getStatus(data.battery_voltage, THRESHOLDS.battery_voltage)))} />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold font-mono">{formatVoltage(data.battery_voltage)}</span>
              <span className="text-porsche-gray-500 text-sm">
                {data.battery_voltage > 13.5 ? 'Charging' : 'On Battery'}
              </span>
            </div>
          </div>

          {/* Vehicle Info - Compact */}
          <div className="bg-porsche-gray-800 rounded-xl border border-porsche-gray-700 p-4 flex-1">
            <span className="text-porsche-gray-400 text-sm font-medium">VEHICLE</span>
            <div className="mt-2 space-y-1.5 text-sm">
              <div className="flex justify-between">
                <span className="text-porsche-gray-500">VIN</span>
                <span className="font-mono text-xs">{vehicle.vin}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-porsche-gray-500">Color</span>
                <span>{vehicle.color}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-porsche-gray-500">Status</span>
                <span className={connectionStatus === 'connected' ? 'text-green-500' : 'text-yellow-500'}>
                  {connectionStatus === 'connected' ? 'Online' : 'Connecting...'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
