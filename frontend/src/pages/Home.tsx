import { useState, useEffect, Suspense, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows, useGLTF, Center, PerspectiveCamera } from '@react-three/drei';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import { Cloud, Sun, CloudRain, Music, Pause, Play, SkipForward, SkipBack, Smartphone } from 'lucide-react';
import { useTelemetry } from '../hooks/useTelemetry';
import { vehicleApi } from '../lib/api';
import { Vehicle } from '../types';
import * as THREE from 'three';

// Orange accent color
const ORANGE = '#f97316';
const ORANGE_DIM = '#f9731640';

// Dark theme colors
const DARK_BG = '#0a0a0a';
const DARK_CARD = '#141414';
const DARK_BORDER = '#262626';
const DARK_TEXT = '#ffffff';
const DARK_TEXT_MUTED = '#737373';

// 3D Porsche Model Component
function PorscheModel() {
  const groupRef = useRef<THREE.Group>(null);
  const { scene } = useGLTF('/models/porsche911.glb');
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 1.5) * 0.02;
    }
  });

  return (
    <group ref={groupRef}>
      <Center>
        <primitive 
          object={scene.clone()} 
          scale={0.8} 
          rotation={[0, Math.PI / 5, 0]} 
        />
      </Center>
    </group>
  );
}

useGLTF.preload('/models/porsche911.glb');

// Loading spinner
function ModelLoader() {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y += delta * 2;
  });
  return (
    <mesh ref={meshRef}>
      <torusGeometry args={[1, 0.2, 16, 32]} />
      <meshStandardMaterial color={ORANGE} wireframe />
    </mesh>
  );
}

// Speed & Gear Overlay
function SpeedGearOverlay({ speed, gear }: { speed: number; gear: number }) {
  const gearStr = gear === -1 ? 'R' : gear === 0 ? 'N' : gear.toString();
  
  return (
    <div 
      className="absolute bottom-4 left-4 backdrop-blur-sm rounded-2xl px-5 py-3 shadow-lg"
      style={{ backgroundColor: 'rgba(20, 20, 20, 0.9)', border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="flex items-end gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider" style={{ color: DARK_TEXT_MUTED }}>Speed</div>
          <div className="text-4xl font-bold" style={{ color: DARK_TEXT, fontFamily: 'monospace' }}>
            {Math.round(speed)}
            <span className="text-lg ml-1" style={{ color: DARK_TEXT_MUTED }}>km/h</span>
          </div>
        </div>
        <div className="pl-4" style={{ borderLeft: `1px solid ${DARK_BORDER}` }}>
          <div className="text-xs uppercase tracking-wider" style={{ color: DARK_TEXT_MUTED }}>Gear</div>
          <div className="text-4xl font-bold" style={{ color: ORANGE, fontFamily: 'monospace' }}>
            {gearStr}
          </div>
        </div>
      </div>
    </div>
  );
}

// Weather Widget
function WeatherWidget() {
  const [weather] = useState({
    temp: 22,
    condition: 'sunny',
    city: 'Stuttgart'
  });
  
  const WeatherIcon = weather.condition === 'sunny' ? Sun : 
                      weather.condition === 'rainy' ? CloudRain : Cloud;
  
  return (
    <div 
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="text-xs uppercase tracking-wider mb-3" style={{ color: DARK_TEXT_MUTED }}>Weather</div>
      <div className="flex items-center gap-4">
        <WeatherIcon className="w-12 h-12" style={{ color: ORANGE }} />
        <div>
          <div className="text-3xl font-bold" style={{ color: DARK_TEXT }}>{weather.temp}°C</div>
          <div className="text-sm capitalize" style={{ color: DARK_TEXT_MUTED }}>{weather.condition}</div>
        </div>
      </div>
      <div className="text-xs mt-3" style={{ color: DARK_TEXT_MUTED }}>{weather.city}</div>
    </div>
  );
}

// Battery Widget - Fixed with proper SVG
function BatteryWidget({ voltage, speed }: { voltage: number; speed: number }) {
  const percentage = Math.min(100, Math.max(0, ((voltage - 11.5) / 3) * 100));
  const isCharging = voltage > 13.5 && speed === 0;
  const fillColor = percentage > 20 ? ORANGE : '#ef4444';
  
  return (
    <div 
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="text-xs uppercase tracking-wider mb-3" style={{ color: DARK_TEXT_MUTED }}>Battery</div>
      <div className="flex items-center gap-4">
        {/* SVG Battery Icon */}
        <svg width="56" height="32" viewBox="0 0 56 32">
          {/* Battery body outline */}
          <rect 
            x="2" y="4" width="44" height="24" rx="4" ry="4" 
            fill="none" 
            stroke={DARK_TEXT_MUTED} 
            strokeWidth="2"
          />
          {/* Battery tip */}
          <rect 
            x="46" y="10" width="6" height="12" rx="2" ry="2" 
            fill={DARK_TEXT_MUTED}
          />
          {/* Battery fill */}
          <rect 
            x="6" y="8" 
            width={Math.max(0, (percentage / 100) * 36)} 
            height="16" 
            rx="2" ry="2" 
            fill={fillColor}
          />
        </svg>
        <div>
          <div className="text-3xl font-bold" style={{ color: DARK_TEXT }}>{percentage.toFixed(0)}%</div>
          <div className="text-sm" style={{ color: DARK_TEXT_MUTED }}>
            {isCharging ? 'Charging' : 'In Use'}
          </div>
        </div>
      </div>
      <div className="text-xs mt-3" style={{ color: DARK_TEXT_MUTED }}>{voltage.toFixed(1)}V</div>
    </div>
  );
}

// Music Player Widget
function MusicWidget() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [connected, setConnected] = useState(false);
  
  if (!connected) {
    return (
      <div 
        className="rounded-2xl p-5 h-full"
        style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
      >
        <div className="text-xs uppercase tracking-wider mb-3" style={{ color: DARK_TEXT_MUTED }}>Music</div>
        <div className="flex flex-col items-center justify-center py-4">
          <Smartphone className="w-10 h-10 mb-3" style={{ color: DARK_TEXT_MUTED }} />
          <div className="text-sm mb-3" style={{ color: DARK_TEXT_MUTED }}>Connect your phone</div>
          <button 
            onClick={() => setConnected(true)}
            className="px-4 py-2 rounded-full text-sm font-medium text-white transition-all hover:opacity-90"
            style={{ backgroundColor: ORANGE }}
          >
            Connect via Bluetooth
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div 
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="text-xs uppercase tracking-wider mb-3" style={{ color: DARK_TEXT_MUTED }}>Now Playing</div>
      <div className="flex items-center gap-4 mb-4">
        <div 
          className="w-14 h-14 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: ORANGE_DIM }}
        >
          <Music className="w-7 h-7" style={{ color: ORANGE }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold truncate" style={{ color: DARK_TEXT }}>Night Dancer</div>
          <div className="text-xs truncate" style={{ color: DARK_TEXT_MUTED }}>Imase</div>
        </div>
      </div>
      <div className="h-1 rounded-full mb-3 overflow-hidden" style={{ backgroundColor: DARK_BORDER }}>
        <div className="h-full rounded-full" style={{ width: '35%', backgroundColor: ORANGE }} />
      </div>
      <div className="flex items-center justify-center gap-4">
        <button className="p-2 transition-colors" style={{ color: DARK_TEXT_MUTED }}>
          <SkipBack className="w-5 h-5" />
        </button>
        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          className="p-3 rounded-full text-white transition-all hover:opacity-90"
          style={{ backgroundColor: ORANGE }}
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
        </button>
        <button className="p-2 transition-colors" style={{ color: DARK_TEXT_MUTED }}>
          <SkipForward className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

// Speed Graph Widget - Fixed with proper chart
function SpeedGraphWidget({ history }: { history: { speed: number; time: string }[] }) {
  // Ensure we have data to display
  const displayData = history.length > 0 ? history : [{ speed: 0, time: '0' }];
  
  return (
    <div 
      className="rounded-2xl p-5"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="text-xs uppercase tracking-wider mb-2" style={{ color: DARK_TEXT_MUTED }}>Speed History</div>
      <div className="h-[100px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={displayData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <YAxis hide domain={[0, 'dataMax + 20']} />
            <Line 
              type="monotone" 
              dataKey="speed" 
              stroke={ORANGE}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-between text-xs mt-1" style={{ color: DARK_TEXT_MUTED }}>
        <span>60s ago</span>
        <span>Now</span>
      </div>
    </div>
  );
}

// Trip Info Widget
function TripInfoWidget({ vehicleId }: { vehicleId: string | null }) {
  const [activeTrip, setActiveTrip] = useState<any>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!vehicleId) return;
    
    const fetchTrip = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/telemetry/trips/active/${vehicleId}`);
        if (res.ok) {
          const trip = await res.json();
          if (trip) setActiveTrip(trip);
        }
      } catch (e) {
        console.error('Failed to fetch trip:', e);
      }
    };
    
    fetchTrip();
    const interval = setInterval(fetchTrip, 5000);
    return () => clearInterval(interval);
  }, [vehicleId]);

  useEffect(() => {
    if (!activeTrip) return;
    const startTime = new Date(activeTrip.start_time).getTime();
    
    const updateElapsed = () => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    };
    
    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);
    return () => clearInterval(interval);
  }, [activeTrip]);

  const formatTime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) return `${hrs}h ${mins}m`;
    return `${mins}m ${secs}s`;
  };

  const formatStartTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  };

  if (!activeTrip) {
    return (
      <div 
        className="rounded-2xl p-5 h-full"
        style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
      >
        <div className="text-xs uppercase tracking-wider mb-3" style={{ color: DARK_TEXT_MUTED }}>Current Trip</div>
        <div className="flex flex-col items-center justify-center py-6">
          <div className="text-4xl mb-2">🚗</div>
          <div className="text-sm" style={{ color: DARK_TEXT_MUTED }}>No active trip</div>
          <div className="text-xs mt-1" style={{ color: DARK_TEXT_MUTED }}>Start driving to begin</div>
        </div>
      </div>
    );
  }

  const avgSpeed = activeTrip.speed_sum && activeTrip.total_readings 
    ? activeTrip.speed_sum / activeTrip.total_readings 
    : 0;
  const distanceKm = (avgSpeed * elapsed) / 3600;

  return (
    <div 
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="text-xs uppercase tracking-wider mb-3" style={{ color: DARK_TEXT_MUTED }}>Current Trip</div>
      <div className="flex items-center gap-2 mb-4">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        <span className="text-sm" style={{ color: DARK_TEXT_MUTED }}>Started at {formatStartTime(activeTrip.start_time)}</span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-3xl font-bold" style={{ color: DARK_TEXT }}>{formatTime(elapsed)}</div>
          <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>Duration</div>
        </div>
        <div>
          <div className="text-3xl font-bold" style={{ color: ORANGE }}>
            {distanceKm.toFixed(1)}
            <span className="text-lg ml-1" style={{ color: DARK_TEXT_MUTED }}>km</span>
          </div>
          <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>Distance</div>
        </div>
      </div>
      <div className="flex gap-4 mt-4 pt-4" style={{ borderTop: `1px solid ${DARK_BORDER}` }}>
        <div className="text-center flex-1">
          <div className="text-lg font-semibold" style={{ color: DARK_TEXT }}>{Math.round(activeTrip.max_speed_kmh || 0)}</div>
          <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>Max km/h</div>
        </div>
        <div className="text-center flex-1">
          <div className="text-lg font-semibold" style={{ color: DARK_TEXT }}>{(activeTrip.max_rpm || 0).toLocaleString()}</div>
          <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>Max RPM</div>
        </div>
      </div>
    </div>
  );
}

// Main Home Component
export default function Home() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);
  const [speedHistory, setSpeedHistory] = useState<{ speed: number; time: string }[]>([]);
  const historyRef = useRef<{ speed: number; time: string }[]>([]);

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
      const now = new Date().toLocaleTimeString();
      const newPoint = { speed: telemetry.speed_kmh, time: now };
      
      if (historyRef.current.length >= 60) {
        historyRef.current.shift();
      }
      historyRef.current.push(newPoint);
      setSpeedHistory([...historyRef.current]);
    }
  }, [telemetry?.speed_kmh]);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center" style={{ backgroundColor: DARK_BG }}>
        <div style={{ color: DARK_TEXT_MUTED }}>Loading...</div>
      </div>
    );
  }

  const data = telemetry ?? {
    speed_kmh: 0,
    rpm: 0,
    gear: 0,
    battery_voltage: 12.6,
  };

  return (
    <div className="h-full p-6 overflow-auto" style={{ backgroundColor: DARK_BG }}>
      {/* Grid Layout */}
      <div className="grid grid-cols-12 gap-5 h-full">
        
        {/* Left Column - 3D Car + Speed History underneath */}
        <div className="col-span-7 flex flex-col gap-5">
          {/* 3D Car */}
          <div 
            className="rounded-2xl relative overflow-hidden flex-1"
            style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
          >
            {/* Connection Status */}
            <div 
              className="absolute top-4 right-4 z-10 flex items-center gap-2 backdrop-blur-sm rounded-full px-3 py-1.5"
              style={{ backgroundColor: 'rgba(20, 20, 20, 0.8)' }}
            >
              <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
                {connectionStatus === 'connected' ? 'Live' : 'Offline'}
              </span>
            </div>
            
            {/* 3D Canvas - Fixed lighting and ground */}
            <Canvas shadows>
              <PerspectiveCamera makeDefault position={[6, 2, 7]} fov={40} />
              <OrbitControls
                enablePan={false}
                enableZoom={true}
                minDistance={5}
                maxDistance={12}
                maxPolarAngle={Math.PI / 2.1}
                minPolarAngle={Math.PI / 6}
                autoRotate
                autoRotateSpeed={0.5}
              />
              
              <Environment preset="studio" />
              
              <Suspense fallback={<ModelLoader />}>
                <PorscheModel />
              </Suspense>
              
              {/* Adjusted ground position to not cut car */}
              <ContactShadows 
                position={[0, -1.2, 0]} 
                opacity={0.5} 
                scale={12} 
                blur={2} 
                far={10}
              />
            </Canvas>
            
            {/* Speed & Gear Overlay */}
            <SpeedGearOverlay speed={data.speed_kmh} gear={data.gear} />
            
            {/* Car Name */}
            <div className="absolute top-4 left-4">
              <div className="text-lg font-bold" style={{ color: DARK_TEXT }}>Porsche 911</div>
              <div className="text-sm" style={{ color: DARK_TEXT_MUTED }}>Turbo S • 2024</div>
            </div>
          </div>
          
          {/* Speed History - Full width under 3D model */}
          <SpeedGraphWidget history={speedHistory} />
        </div>
        
        {/* Right Column */}
        <div className="col-span-5 flex flex-col gap-5">
          {/* Top Row - Weather & Battery */}
          <div className="grid grid-cols-2 gap-5">
            <WeatherWidget />
            <BatteryWidget voltage={data.battery_voltage} speed={data.speed_kmh} />
          </div>
          
          {/* Music Player */}
          <MusicWidget />
          
          {/* Current Trip */}
          <TripInfoWidget vehicleId={vehicle?.id ?? null} />
        </div>
        
      </div>
    </div>
  );
}
