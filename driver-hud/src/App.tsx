import { useState, useEffect, useRef } from 'react';
import { useTelemetry } from './hooks/useTelemetry';

// Orange theme colors
const ORANGE = '#f97316';
const ORANGE_DIM = '#f9731630';

// Alert thresholds
const THRESHOLDS = {
  engine_temp: { warning: 100, critical: 110 },
  oil_temp: { warning: 120, critical: 140 },
  fuel_level: { warning: 15, critical: 5 },
  battery_voltage: { low: 12.2, critical: 11.8 },
  rpm: { warning: 7000, redline: 7500 },
  tire_pressure: { low: 30, critical: 28 },
};

// Fetch vehicle ID from API
async function fetchVehicleId(): Promise<string | null> {
  try {
    const res = await fetch('http://localhost:8000/api/telemetry/vehicles');
    if (!res.ok) throw new Error('Failed to fetch');
    const vehicles = await res.json();
    return vehicles[0]?.id || null;
  } catch (e) {
    console.error('Error fetching vehicle:', e);
    return null;
  }
}

// ============================================
// SMOOTH VALUE HOOK
// ============================================
function useSmoothValue(target: number, smoothing: number = 0.1) {
  const [value, setValue] = useState(target);
  const animationRef = useRef<number>();

  useEffect(() => {
    const animate = () => {
      setValue(prev => {
        const diff = target - prev;
        if (Math.abs(diff) < 0.5) return target;
        return prev + diff * smoothing;
      });
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [target, smoothing]);

  return value;
}

// ============================================
// ALERT ICON COMPONENT
// ============================================
interface AlertIconProps {
  type: 'engine' | 'oil' | 'fuel' | 'battery' | 'tire' | 'rpm';
  active: boolean;
  critical?: boolean;
}

function AlertIcon({ type, active, critical = false }: AlertIconProps) {
  if (!active) return null;
  
  const color = critical ? '#ef4444' : '#f59e0b';
  
  const icons: Record<string, JSX.Element> = {
    engine: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill={color}>
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
      </svg>
    ),
    oil: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill={color}>
        <path d="M19 14V6c0-1.1-.9-2-2-2H7c-1.1 0-2 .9-2 2v8c0 2.21 1.79 4 4 4h1v2H8v2h8v-2h-2v-2h1c2.21 0 4-1.79 4-4zM8 8h8v2H8V8z"/>
      </svg>
    ),
    fuel: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill={color}>
        <path d="M19.77 7.23l.01-.01-3.72-3.72L15 4.56l2.11 2.11c-.94.36-1.61 1.26-1.61 2.33 0 1.38 1.12 2.5 2.5 2.5.36 0 .69-.08 1-.21v7.21c0 .55-.45 1-1 1s-1-.45-1-1V14c0-1.1-.9-2-2-2h-1V5c0-1.1-.9-2-2-2H6c-1.1 0-2 .9-2 2v16h10v-7.5h1.5v5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V9c0-.69-.28-1.32-.73-1.77zM12 10H6V5h6v5z"/>
      </svg>
    ),
    battery: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill={color}>
        <path d="M15.67 4H14V2h-4v2H8.33C7.6 4 7 4.6 7 5.33v15.33C7 21.4 7.6 22 8.33 22h7.33c.74 0 1.34-.6 1.34-1.33V5.33C17 4.6 16.4 4 15.67 4z"/>
      </svg>
    ),
    tire: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill={color}>
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6z"/>
      </svg>
    ),
    rpm: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill={color}>
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
      </svg>
    ),
  };

  return (
    <div 
      className={`${critical ? 'animate-pulse' : ''}`}
      title={`${type.toUpperCase()} ${critical ? 'CRITICAL' : 'WARNING'}`}
    >
      {icons[type]}
    </div>
  );
}

// ============================================
// FUEL GAUGE
// ============================================
function FuelGauge({ level }: { level: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothLevel = useSmoothValue(level, 0.1);
  const isLow = level < 15;
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const size = canvas.width;
    const center = size / 2;
    const radius = size / 2 - 12;
    
    ctx.clearRect(0, 0, size, size);
    
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const angleRange = endAngle - startAngle;
    
    ctx.beginPath();
    ctx.arc(center, center, radius, startAngle, endAngle);
    ctx.strokeStyle = ORANGE_DIM;
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    const fuelAngle = startAngle + (smoothLevel / 100) * angleRange;
    ctx.beginPath();
    ctx.arc(center, center, radius, startAngle, fuelAngle);
    ctx.strokeStyle = isLow ? '#ef4444' : ORANGE;
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    ctx.font = '10px monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'center';
    const labelR = radius - 18;
    ctx.fillText('E', center + Math.cos(startAngle) * labelR, center + Math.sin(startAngle) * labelR);
    ctx.fillText('F', center + Math.cos(endAngle) * labelR, center + Math.sin(endAngle) * labelR);
    
  }, [smoothLevel, isLow]);
  
  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-600 mb-1 tracking-widest">FUEL</div>
      <div className="relative">
        <canvas ref={canvasRef} width={90} height={90} />
        <div className="absolute inset-0 flex items-center justify-center">
          <span 
            className="text-lg font-bold font-mono"
            style={{ color: isLow ? '#ef4444' : ORANGE }}
          >
            {Math.round(smoothLevel)}%
          </span>
        </div>
      </div>
    </div>
  );
}

// ============================================
// RPM GAUGE
// ============================================
function RpmGauge({ rpm, maxRpm = 8000 }: { rpm: number; maxRpm?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothRpm = useSmoothValue(rpm, 0.15);
  const isHigh = rpm >= 7000;
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const size = canvas.width;
    const center = size / 2;
    const radius = size / 2 - 20;
    
    ctx.clearRect(0, 0, size, size);
    
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const angleRange = endAngle - startAngle;
    
    ctx.beginPath();
    ctx.arc(center, center, radius, startAngle, endAngle);
    ctx.strokeStyle = ORANGE_DIM;
    ctx.lineWidth = 3;
    ctx.stroke();
    
    const markers = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    markers.forEach((val) => {
      const angle = startAngle + (val / 8) * angleRange;
      const isRedline = val >= 7;
      
      const innerR = radius - 12;
      const outerR = radius - 2;
      ctx.beginPath();
      ctx.moveTo(center + Math.cos(angle) * innerR, center + Math.sin(angle) * innerR);
      ctx.lineTo(center + Math.cos(angle) * outerR, center + Math.sin(angle) * outerR);
      ctx.strokeStyle = isRedline ? '#ef4444' : ORANGE;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      const textR = radius + 12;
      ctx.font = 'bold 11px monospace';
      ctx.fillStyle = isRedline ? '#ef4444' : ORANGE;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(val.toString(), center + Math.cos(angle) * textR, center + Math.sin(angle) * textR);
    });
    
    ctx.font = '10px monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'center';
    ctx.fillText('RPM', center, center - 30);
    
    const rpmAngle = startAngle + (Math.min(smoothRpm, maxRpm) / maxRpm) * angleRange;
    const needleLength = radius - 25;
    
    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.lineTo(center + Math.cos(rpmAngle) * needleLength, center + Math.sin(rpmAngle) * needleLength);
    ctx.strokeStyle = isHigh ? '#ef4444' : ORANGE;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(center, center, 8, 0, Math.PI * 2);
    ctx.fillStyle = isHigh ? '#ef4444' : ORANGE;
    ctx.fill();
    
  }, [smoothRpm, isHigh, maxRpm]);
  
  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-600 mb-1 tracking-widest">RPM</div>
      <div className="relative">
        <canvas ref={canvasRef} width={200} height={200} />
        <div className="absolute inset-0 flex items-center justify-center pt-16">
          <span 
            className="text-3xl font-bold font-mono"
            style={{ color: isHigh ? '#ef4444' : ORANGE }}
          >
            {Math.round(smoothRpm).toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}

// ============================================
// SPEEDOMETER
// ============================================
function Speedometer({ speed, maxSpeed = 320 }: { speed: number; maxSpeed?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothSpeed = useSmoothValue(speed, 0.12);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const size = canvas.width;
    const center = size / 2;
    const radius = size / 2 - 25;
    
    ctx.clearRect(0, 0, size, size);
    
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const angleRange = endAngle - startAngle;
    
    ctx.beginPath();
    ctx.arc(center, center, radius + 10, startAngle, endAngle);
    ctx.strokeStyle = ORANGE_DIM;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    const markers = [0, 40, 80, 120, 160, 200, 240, 280, 320];
    markers.forEach((val) => {
      const angle = startAngle + (val / maxSpeed) * angleRange;
      
      const innerR = radius - 10;
      const outerR = radius;
      ctx.beginPath();
      ctx.moveTo(center + Math.cos(angle) * innerR, center + Math.sin(angle) * innerR);
      ctx.lineTo(center + Math.cos(angle) * outerR, center + Math.sin(angle) * outerR);
      ctx.strokeStyle = ORANGE;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      const textR = radius + 22;
      ctx.font = 'bold 14px monospace';
      ctx.fillStyle = ORANGE;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(val.toString(), center + Math.cos(angle) * textR, center + Math.sin(angle) * textR);
    });
    
    for (let val = 0; val <= maxSpeed; val += 10) {
      if (val % 40 === 0) continue;
      const angle = startAngle + (val / maxSpeed) * angleRange;
      const innerR = radius - 5;
      const outerR = radius;
      
      ctx.beginPath();
      ctx.moveTo(center + Math.cos(angle) * innerR, center + Math.sin(angle) * innerR);
      ctx.lineTo(center + Math.cos(angle) * outerR, center + Math.sin(angle) * outerR);
      ctx.strokeStyle = ORANGE_DIM;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    
    ctx.font = '12px monospace';
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';
    ctx.fillText('km/h', center, center - 40);
    
    const speedAngle = startAngle + (Math.min(smoothSpeed, maxSpeed) / maxSpeed) * angleRange;
    const needleLength = radius - 25;
    
    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.lineTo(center + Math.cos(speedAngle) * needleLength, center + Math.sin(speedAngle) * needleLength);
    ctx.strokeStyle = ORANGE;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(center, center, 10, 0, Math.PI * 2);
    ctx.fillStyle = ORANGE;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    
  }, [smoothSpeed, maxSpeed]);
  
  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-600 mb-1 tracking-widest">SPEED</div>
      <div className="relative">
        <canvas ref={canvasRef} width={320} height={320} />
        <div className="absolute inset-0 flex items-center justify-center pt-24">
          <span className="text-7xl font-bold font-mono" style={{ color: ORANGE }}>
            {Math.round(smoothSpeed)}
          </span>
        </div>
      </div>
    </div>
  );
}

// ============================================
// MAP WITH ARROW
// ============================================
function MapDisplay({ lat, lng, heading }: { lat: number; lng: number; heading: number }) {
  const mapUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${lng-0.006}%2C${lat-0.005}%2C${lng+0.006}%2C${lat+0.005}&layer=mapnik`;
  
  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-600 mb-1 tracking-widest">MAP</div>
      <div 
        className="relative overflow-hidden rounded-full"
        style={{ 
          width: 200, 
          height: 200, 
          border: `3px solid ${ORANGE}`,
          boxShadow: `0 0 20px ${ORANGE}40`
        }}
      >
        <iframe
          src={mapUrl}
          width="300"
          height="300"
          style={{ 
            border: 0,
            marginLeft: -50,
            marginTop: -50,
            filter: 'grayscale(100%) invert(92%) contrast(1.1) brightness(0.9)',
            pointerEvents: 'none',
          }}
          title="Map"
        />
        
        <div 
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
          style={{ transform: `rotate(${heading}deg)` }}
        >
          <svg width="60" height="60" viewBox="0 0 60 60">
            <defs>
              <filter id="arrowGlow" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feFlood floodColor={ORANGE} floodOpacity="0.6"/>
                <feComposite in2="blur" operator="in"/>
                <feMerge>
                  <feMergeNode/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>
            <g filter="url(#arrowGlow)">
              <path 
                d="M30 6 L42 50 L30 40 L18 50 Z" 
                fill={ORANGE}
                stroke="white"
                strokeWidth="1.5"
              />
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}

// ============================================
// GEAR DISPLAY
// ============================================
function GearDisplay({ gear }: { gear: number }) {
  const gearStr = gear === -1 ? 'R' : gear === 0 ? 'N' : gear.toString();
  const isReverse = gear === -1;
  
  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-600 mb-1 tracking-widest">GEAR</div>
      <div 
        className="w-[90px] h-[90px] rounded-full flex items-center justify-center"
        style={{ 
          border: `3px solid ${isReverse ? '#a855f7' : ORANGE}`,
          boxShadow: `0 0 15px ${isReverse ? '#a855f740' : ORANGE + '40'}`
        }}
      >
        <span 
          className="text-5xl font-bold font-mono"
          style={{ color: isReverse ? '#a855f7' : ORANGE }}
        >
          {gearStr}
        </span>
      </div>
      <div className="flex gap-1 mt-2">
        {['R', 'N', '1', '2', '3', '4', '5', '6', '7'].map((g) => {
          const gearNum = g === 'R' ? -1 : g === 'N' ? 0 : parseInt(g);
          const isActive = gear === gearNum;
          return (
            <div 
              key={g}
              className="w-2 h-2 rounded-full transition-all duration-200"
              style={{ 
                backgroundColor: isActive 
                  ? (gearNum === -1 ? '#a855f7' : ORANGE)
                  : '#333',
                boxShadow: isActive ? `0 0 8px ${gearNum === -1 ? '#a855f7' : ORANGE}` : 'none'
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

// ============================================
// MAIN APP
// ============================================
export default function App() {
  const [vehicleId, setVehicleId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadVehicle = async () => {
      const id = await fetchVehicleId();
      setVehicleId(id);
      setLoading(false);
    };
    loadVehicle();
    
    const interval = setInterval(async () => {
      if (!vehicleId) {
        const id = await fetchVehicleId();
        if (id) setVehicleId(id);
      }
    }, 3000);
    
    return () => clearInterval(interval);
  }, [vehicleId]);

  const { telemetry, connectionStatus } = useTelemetry({
    vehicleId,
    enabled: !!vehicleId,
  });

  const data = telemetry ?? {
    speed_kmh: 0,
    rpm: 800,
    gear: 0,
    fuel_level: 85,
    engine_temp: 20,
    oil_temp: 20,
    battery_voltage: 12.6,
    latitude: 48.8342,
    longitude: 9.1519,
    heading: 0,
    tire_pressure_fl: 33,
    tire_pressure_fr: 33,
    tire_pressure_rl: 32,
    tire_pressure_rr: 32,
  };

  // Check for alerts
  const alerts = {
    engine: data.engine_temp >= THRESHOLDS.engine_temp.warning,
    engineCritical: data.engine_temp >= THRESHOLDS.engine_temp.critical,
    oil: data.oil_temp >= THRESHOLDS.oil_temp.warning,
    oilCritical: data.oil_temp >= THRESHOLDS.oil_temp.critical,
    fuel: data.fuel_level <= THRESHOLDS.fuel_level.warning,
    fuelCritical: data.fuel_level <= THRESHOLDS.fuel_level.critical,
    battery: data.battery_voltage <= THRESHOLDS.battery_voltage.low,
    batteryCritical: data.battery_voltage <= THRESHOLDS.battery_voltage.critical,
    rpm: data.rpm >= THRESHOLDS.rpm.warning,
    rpmCritical: data.rpm >= THRESHOLDS.rpm.redline,
    tire: (data.tire_pressure_fl <= THRESHOLDS.tire_pressure.low) ||
          (data.tire_pressure_fr <= THRESHOLDS.tire_pressure.low) ||
          (data.tire_pressure_rl <= THRESHOLDS.tire_pressure.low) ||
          (data.tire_pressure_rr <= THRESHOLDS.tire_pressure.low),
    tireCritical: (data.tire_pressure_fl <= THRESHOLDS.tire_pressure.critical) ||
                  (data.tire_pressure_fr <= THRESHOLDS.tire_pressure.critical) ||
                  (data.tire_pressure_rl <= THRESHOLDS.tire_pressure.critical) ||
                  (data.tire_pressure_rr <= THRESHOLDS.tire_pressure.critical),
  };

  const hasAnyAlert = alerts.engine || alerts.oil || alerts.fuel || alerts.battery || alerts.rpm || alerts.tire;

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      {/* Main HUD Container */}
      <div 
        className="rounded-2xl p-6 relative"
        style={{ 
          background: 'linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%)',
          border: '1px solid #333',
          boxShadow: `0 0 60px ${ORANGE}15, inset 0 0 30px rgba(0,0,0,0.5)`
        }}
      >
        {/* Alert Icons - TOP LEFT inside card */}
        <div className="absolute top-4 left-4 flex gap-2">
          <AlertIcon type="fuel" active={alerts.fuel} critical={alerts.fuelCritical} />
          <AlertIcon type="engine" active={alerts.engine} critical={alerts.engineCritical} />
          <AlertIcon type="oil" active={alerts.oil} critical={alerts.oilCritical} />
        </div>

        {/* Alert Icons - TOP RIGHT inside card */}
        <div className="absolute top-4 right-4 flex gap-2">
          <AlertIcon type="battery" active={alerts.battery} critical={alerts.batteryCritical} />
          <AlertIcon type="tire" active={alerts.tire} critical={alerts.tireCritical} />
          <AlertIcon type="rpm" active={alerts.rpm} critical={alerts.rpmCritical} />
          
          {/* Connection Status */}
          <div className="flex items-center gap-1.5 ml-2 px-2 py-1 rounded-full bg-gray-900/50">
            <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-[10px] text-gray-400 uppercase tracking-wider">
              {connectionStatus === 'connected' ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Gauges Row */}
        <div className="flex items-end gap-4 mt-6">
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
    </div>
  );
}
