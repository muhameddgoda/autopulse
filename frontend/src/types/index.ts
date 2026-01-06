// Vehicle types
export interface Vehicle {
  id: string;
  vin: string;
  make: string;
  model: string;
  variant: string | null;
  year: number | null;
  color: string | null;
  created_at: string;
  updated_at: string;
}

// Driving mode type
export type DrivingMode = 'parked' | 'reverse' | 'city' | 'highway' | 'sport';

// Telemetry reading from API/WebSocket
export interface TelemetryReading {
  time: string;
  vehicle_id: string;
  speed_kmh: number;
  rpm: number;
  gear: number;
  throttle_position: number;
  engine_temp: number;
  oil_temp: number;
  oil_pressure: number;
  fuel_level: number;
  battery_voltage: number;
  tire_pressure_fl: number;
  tire_pressure_fr: number;
  tire_pressure_rl: number;
  tire_pressure_rr: number;
  latitude: number | null;
  longitude: number | null;
  heading: number | null;
  driving_mode: DrivingMode | null;
}

// Mode theme configuration
export interface ModeTheme {
  name: string;
  icon: string;
  primaryColor: string;
  secondaryColor: string;
  accentColor: string;
  bgGradient: string;
  glowColor: string;
}

export const MODE_THEMES: Record<DrivingMode, ModeTheme> = {
  parked: {
    name: 'PARKED',
    icon: '🅿️',
    primaryColor: '#6b7280',
    secondaryColor: '#374151',
    accentColor: '#9ca3af',
    bgGradient: 'from-gray-900 to-gray-950',
    glowColor: 'rgba(107, 114, 128, 0.3)',
  },
  reverse: {
    name: 'REVERSE',
    icon: '🔄',
    primaryColor: '#a855f7',
    secondaryColor: '#7c3aed',
    accentColor: '#c084fc',
    bgGradient: 'from-purple-950 to-gray-950',
    glowColor: 'rgba(168, 85, 247, 0.3)',
  },
  city: {
    name: 'CITY',
    icon: '🏙️',
    primaryColor: '#06b6d4',
    secondaryColor: '#0891b2',
    accentColor: '#22d3ee',
    bgGradient: 'from-cyan-950 to-gray-950',
    glowColor: 'rgba(6, 182, 212, 0.3)',
  },
  highway: {
    name: 'HIGHWAY',
    icon: '🛣️',
    primaryColor: '#3b82f6',
    secondaryColor: '#1d4ed8',
    accentColor: '#60a5fa',
    bgGradient: 'from-blue-950 to-gray-950',
    glowColor: 'rgba(59, 130, 246, 0.3)',
  },
  sport: {
    name: 'SPORT',
    icon: '🔥',
    primaryColor: '#f97316',
    secondaryColor: '#ea580c',
    accentColor: '#fb923c',
    bgGradient: 'from-orange-950 to-red-950',
    glowColor: 'rgba(249, 115, 22, 0.4)',
  },
};

// WebSocket message types
export type WebSocketMessageType = 
  | 'connected'
  | 'telemetry_update'
  | 'trip_started'
  | 'trip_ended'
  | 'keepalive'
  | 'pong';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  data?: TelemetryReading | Record<string, unknown>;
  vehicle_id?: string;
  message?: string;
}

// Trip types
export interface Trip {
  id: string;
  vehicle_id: string;
  start_time: string;
  end_time: string | null;
  is_active: boolean;
  
  // Core stats
  distance_km: number | null;
  duration_seconds: number | null;
  
  // Speed stats
  avg_speed_kmh: number | null;
  max_speed_kmh: number | null;
  
  // RPM stats
  avg_rpm: number | null;
  max_rpm: number | null;
  
  // Fuel stats
  fuel_start: number | null;
  fuel_end: number | null;
  fuel_used_liters: number | null;
  
  // Mode breakdown (seconds)
  mode_parked_seconds: number | null;
  mode_city_seconds: number | null;
  mode_highway_seconds: number | null;
  mode_sport_seconds: number | null;
  mode_reverse_seconds: number | null;
  
  // Location
  start_latitude: number | null;
  start_longitude: number | null;
  end_latitude: number | null;
  end_longitude: number | null;
  
  created_at: string;
}

// Dashboard state
export interface DashboardState {
  vehicle: Vehicle | null;
  telemetry: TelemetryReading | null;
  activeTrip: Trip | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastUpdate: Date | null;
}

// Gear display helper
export const getGearDisplay = (gear: number): string => {
  if (gear === -1) return 'R';
  if (gear === 0) return 'N';
  return gear.toString();
};

// Status thresholds
export const THRESHOLDS = {
  engine_temp: { warning: 100, critical: 110 },
  oil_temp: { warning: 120, critical: 140 },
  oil_pressure: { low: 1.5, critical: 1.0 },
  fuel_level: { warning: 15, critical: 5 },
  battery_voltage: { low: 12.2, critical: 11.8 },
  rpm: { warning: 7000, redline: 7500 },
  tire_pressure: { low: 30, critical: 28, high: 38 },
};

export const getStatus = (
  value: number,
  thresholds: { warning?: number; critical?: number; low?: number; redline?: number }
): 'normal' | 'warning' | 'critical' => {
  if (thresholds.critical !== undefined && value >= thresholds.critical) return 'critical';
  if (thresholds.redline !== undefined && value >= thresholds.redline) return 'critical';
  if (thresholds.warning !== undefined && value >= thresholds.warning) return 'warning';
  if (thresholds.low !== undefined && value <= thresholds.low) return 'warning';
  return 'normal';
};
