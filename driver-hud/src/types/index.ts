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
}

export type DrivingMode = 'parked' | 'reverse' | 'city' | 'highway' | 'sport';

export interface WebSocketMessage {
  type: string;
  data?: TelemetryReading | Record<string, unknown>;
  vehicle_id?: string;
  message?: string;
}

export const getGearDisplay = (gear: number): string => {
  if (gear === -1) return 'R';
  if (gear === 0) return 'N';
  return gear.toString();
};

// Mode theme configurations
export interface ModeTheme {
  name: string;
  bgGradient: string;
  primaryColor: string;
  secondaryColor: string;
  accentColor: string;
  glowColor: string;
  textColor: string;
  icon: string;
}

export const MODE_THEMES: Record<DrivingMode, ModeTheme> = {
  parked: {
    name: 'PARKED',
    bgGradient: 'from-gray-950 via-gray-900 to-gray-950',
    primaryColor: '#6b7280',
    secondaryColor: '#374151',
    accentColor: '#9ca3af',
    glowColor: 'rgba(107, 114, 128, 0.3)',
    textColor: '#9ca3af',
    icon: '🅿️',
  },
  reverse: {
    name: 'REVERSE',
    bgGradient: 'from-purple-950 via-gray-900 to-purple-950',
    primaryColor: '#a855f7',
    secondaryColor: '#7c3aed',
    accentColor: '#c084fc',
    glowColor: 'rgba(168, 85, 247, 0.3)',
    textColor: '#c084fc',
    icon: '🔄',
  },
  city: {
    name: 'CITY',
    bgGradient: 'from-cyan-950 via-gray-900 to-teal-950',
    primaryColor: '#06b6d4',
    secondaryColor: '#0891b2',
    accentColor: '#22d3ee',
    glowColor: 'rgba(6, 182, 212, 0.3)',
    textColor: '#22d3ee',
    icon: '🏙️',
  },
  highway: {
    name: 'HIGHWAY',
    bgGradient: 'from-blue-950 via-gray-900 to-slate-950',
    primaryColor: '#3b82f6',
    secondaryColor: '#1d4ed8',
    accentColor: '#60a5fa',
    glowColor: 'rgba(59, 130, 246, 0.3)',
    textColor: '#60a5fa',
    icon: '🛣️',
  },
  sport: {
    name: 'SPORT',
    bgGradient: 'from-orange-950 via-red-950 to-orange-950',
    primaryColor: '#f97316',
    secondaryColor: '#ea580c',
    accentColor: '#fb923c',
    glowColor: 'rgba(249, 115, 22, 0.4)',
    textColor: '#fb923c',
    icon: '🔥',
  },
};

// Warning thresholds
export const WARNINGS = {
  fuel_low: 15,
  fuel_critical: 5,
  rpm_warning: 7000,
  rpm_redline: 7500,
  oil_pressure_low: 1.5,
  oil_pressure_critical: 1.0,
  engine_temp_warning: 105,
  engine_temp_critical: 115,
};
