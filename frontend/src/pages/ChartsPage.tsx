import { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useTelemetry } from '../hooks/useTelemetry';
import { vehicleApi } from '../lib/api';
import { Vehicle } from '../types';

// Theme colors
const ORANGE = '#f97316';
const DARK_BG = '#0a0a0a';
const DARK_CARD = '#141414';
const DARK_BORDER = '#262626';
const DARK_TEXT = '#ffffff';
const DARK_TEXT_MUTED = '#737373';

const MAX_HISTORY = 60;

interface ChartData {
  time: string;
  speed: number;
  rpm: number;
  engineTemp: number;
  throttle: number;
}

// Global history storage
const globalHistory: ChartData[] = [];

// Custom Tooltip
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  
  return (
    <div 
      className="rounded-lg p-3 shadow-xl"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <p className="text-xs mb-2" style={{ color: DARK_TEXT_MUTED }}>{label}</p>
      {payload.map((entry: any, index: number) => (
        <p key={index} className="text-sm font-mono" style={{ color: entry.color }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
        </p>
      ))}
    </div>
  );
}

export default function ChartsPage() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState<ChartData[]>(globalHistory);
  const lastUpdateRef = useRef<string>('');

  useEffect(() => {
    async function fetchVehicle() {
      try {
        const vehicles = await vehicleApi.getAll();
        if (vehicles.length > 0) setVehicle(vehicles[0]);
      } catch (error) {
        console.error('Failed to fetch vehicle:', error);
      } finally {
        setLoading(false);
      }
    }
    fetchVehicle();
  }, []);

  const { telemetry, connectionStatus } = useTelemetry({
    vehicleId: vehicle?.id ?? null,
    enabled: !!vehicle,
  });

  useEffect(() => {
    if (telemetry) {
      const timeStr = new Date().toLocaleTimeString('en-US', { 
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' 
      });
      
      if (timeStr === lastUpdateRef.current) return;
      lastUpdateRef.current = timeStr;
      
      const newPoint: ChartData = {
        time: timeStr,
        speed: telemetry.speed_kmh,
        rpm: telemetry.rpm,
        engineTemp: telemetry.engine_temp,
        throttle: telemetry.throttle_position,
      };
      
      if (globalHistory.length >= MAX_HISTORY) {
        globalHistory.shift();
      }
      globalHistory.push(newPoint);
      setHistory([...globalHistory]);
    }
  }, [telemetry?.speed_kmh]);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center" style={{ backgroundColor: DARK_BG }}>
        <div style={{ color: DARK_TEXT_MUTED }}>Loading...</div>
      </div>
    );
  }

  const displayData = history.length > 0 ? history : [{ time: '0', speed: 0, rpm: 0, engineTemp: 0, throttle: 0 }];

  return (
    <div className="h-full p-6 overflow-auto" style={{ backgroundColor: DARK_BG }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: DARK_TEXT }}>Live Charts</h1>
          <p className="text-sm" style={{ color: DARK_TEXT_MUTED }}>Real-time telemetry visualization</p>
        </div>
        <div 
          className="flex items-center gap-2 rounded-full px-3 py-1.5"
          style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
        >
          <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
            {connectionStatus === 'connected' ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-2 gap-5">
        {/* Speed Chart */}
        <div 
          className="rounded-2xl p-5"
          style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
        >
          <div className="text-xs uppercase tracking-wider mb-4" style={{ color: DARK_TEXT_MUTED }}>
            Speed (km/h)
          </div>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={displayData}>
                <defs>
                  <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={ORANGE} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={ORANGE} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={DARK_BORDER} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} domain={[0, 'dataMax + 20']} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="speed" stroke={ORANGE} fill="url(#speedGrad)" strokeWidth={2} name="Speed" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* RPM Chart */}
        <div 
          className="rounded-2xl p-5"
          style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
        >
          <div className="text-xs uppercase tracking-wider mb-4" style={{ color: DARK_TEXT_MUTED }}>
            RPM
          </div>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={displayData}>
                <defs>
                  <linearGradient id="rpmGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={DARK_BORDER} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} domain={[0, 8000]} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="rpm" stroke="#ef4444" fill="url(#rpmGrad)" strokeWidth={2} name="RPM" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Engine Temp Chart */}
        <div 
          className="rounded-2xl p-5"
          style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
        >
          <div className="text-xs uppercase tracking-wider mb-4" style={{ color: DARK_TEXT_MUTED }}>
            Engine Temperature (°C)
          </div>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayData}>
                <CartesianGrid strokeDasharray="3 3" stroke={DARK_BORDER} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} domain={[0, 120]} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="engineTemp" stroke="#f59e0b" strokeWidth={2} dot={false} name="Engine Temp" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Throttle Chart */}
        <div 
          className="rounded-2xl p-5"
          style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
        >
          <div className="text-xs uppercase tracking-wider mb-4" style={{ color: DARK_TEXT_MUTED }}>
            Throttle Position (%)
          </div>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={displayData}>
                <defs>
                  <linearGradient id="throttleGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={DARK_BORDER} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} axisLine={false} tickLine={false} domain={[0, 100]} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="throttle" stroke="#8b5cf6" fill="url(#throttleGrad)" strokeWidth={2} name="Throttle" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
