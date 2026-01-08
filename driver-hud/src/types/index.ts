// ============================================
// AUTOPULSE TYPE DEFINITIONS
// ============================================

export interface Vehicle {
  id: string;
  vin: string;
  make: string;
  model: string;
  variant: string | null;
  year: number | null;
  color: string | null;
}

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
  // ML metrics
  acceleration_g?: number;
  is_harsh_braking?: boolean;
  is_harsh_acceleration?: boolean;
  engine_stress_score?: number;
}

export type DrivingMode = 'parked' | 'reverse' | 'city' | 'highway' | 'sport';

export interface WebSocketMessage {
  type: string;
  data?: TelemetryReading | Record<string, unknown>;
  vehicle_id?: string;
  message?: string;
}

// Sensor data point for charts
export interface SensorDataPoint {
  time: number;
  value: number;
}

export interface SensorHistory {
  engine_temp: SensorDataPoint[];
  oil_temp: SensorDataPoint[];
  oil_pressure: SensorDataPoint[];
  battery_voltage: SensorDataPoint[];
  fuel_level: SensorDataPoint[];
}

// Warning thresholds
export const THRESHOLDS = {
  engine_temp: { warning: 100, critical: 110, min: 0, max: 130 },
  oil_temp: { warning: 120, critical: 140, min: 0, max: 160 },
  oil_pressure: { low: 1.5, critical: 1.0, min: 0, max: 6 },
  fuel_level: { warning: 15, critical: 5, min: 0, max: 100 },
  battery_voltage: { low: 12.2, critical: 11.8, min: 10, max: 15 },
  rpm: { warning: 7000, redline: 7500, min: 0, max: 8000 },
  tire_pressure: { low: 30, critical: 28, min: 20, max: 40 },
} as const;

// Theme colors
export const THEME = {
  orange: '#f97316',
  orangeDim: '#f9731630',
  orangeGlow: '#f9731640',
  red: '#ef4444',
  redDim: '#ef444430',
  yellow: '#f59e0b',
  green: '#22c55e',
  purple: '#a855f7',
  background: '#0d0d0d',
  cardBg: '#1a1a1a',
  border: '#333',
  textDim: '#666',
  textMuted: '#999',
} as const;

// Helper functions
export const getGearDisplay = (gear: number): string => {
  if (gear === -1) return 'R';
  if (gear === 0) return 'N';
  return gear.toString();
};

export const formatNumber = (value: number, decimals: number = 0): string => {
  return value.toFixed(decimals);
};
