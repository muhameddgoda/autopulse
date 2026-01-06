import { Vehicle, TelemetryReading, Trip } from '../types';

const API_BASE = 'http://localhost:8000/api';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`API Error: ${response.status} - ${errorText}`);
    throw new ApiError(response.status, `HTTP ${response.status}: ${response.statusText}`);
  }

  // Handle empty responses
  const text = await response.text();
  if (!text) return null as T;
  
  try {
    return JSON.parse(text);
  } catch {
    return null as T;
  }
}

// Vehicle endpoints
export const vehicleApi = {
  getAll: () => fetchJson<Vehicle[]>(`${API_BASE}/telemetry/vehicles`),
  
  getById: (id: string) => fetchJson<Vehicle>(`${API_BASE}/telemetry/vehicles/${id}`),
};

// Telemetry endpoints
export const telemetryApi = {
  getLatest: (vehicleId: string) => 
    fetchJson<TelemetryReading | null>(`${API_BASE}/telemetry/latest/${vehicleId}`),
  
  getHistory: (vehicleId: string, minutes: number = 5) =>
    fetchJson<TelemetryReading[]>(`${API_BASE}/telemetry/history/${vehicleId}?minutes=${minutes}`),

  async getDriverSummary(vehicleId: string, days: number) {
    const res = await fetch(`http://localhost:8000/api/telemetry/ml/summary/${vehicleId}?days=${days}`);
    if (!res.ok) throw new Error('Failed to fetch driver summary');
    return res.json();
  },
};

// Trip endpoints
export const tripApi = {
  getActive: async (vehicleId: string): Promise<Trip | null> => {
    try {
      return await fetchJson<Trip | null>(`${API_BASE}/telemetry/trips/active/${vehicleId}`);
    } catch (e) {
      console.error('getActive error:', e);
      return null;
    }
  },
  
  getHistory: async (vehicleId: string, limit: number = 10): Promise<Trip[]> => {
    try {
      return await fetchJson<Trip[]>(`${API_BASE}/telemetry/trips/${vehicleId}?limit=${limit}`);
    } catch (e) {
      console.error('getHistory error:', e);
      return [];
    }
  },
  
  start: async (vehicleId: string, latitude?: number, longitude?: number): Promise<Trip> => {
    console.log('Starting trip for vehicle:', vehicleId);
    const result = await fetchJson<Trip>(`${API_BASE}/telemetry/trips/start`, {
      method: 'POST',
      body: JSON.stringify({
        vehicle_id: vehicleId,
        start_latitude: latitude,
        start_longitude: longitude,
      }),
    });
    console.log('Trip started:', result);
    return result;
  },
  
  end: (tripId: string, latitude?: number, longitude?: number) =>
    fetchJson<Trip>(`${API_BASE}/telemetry/trips/${tripId}/end`, {
      method: 'POST',
      body: JSON.stringify({
        end_latitude: latitude,
        end_longitude: longitude,
      }),
    }),
  
  endActiveTrip: async (vehicleId: string): Promise<Trip | null> => {
    console.log('Ending active trip for vehicle:', vehicleId);
    try {
      const result = await fetchJson<Trip | null>(`${API_BASE}/telemetry/trips/end-active/${vehicleId}`, {
        method: 'POST',
      });
      console.log('Trip ended:', result);
      return result;
    } catch (e) {
      console.error('endActiveTrip error:', e);
      return null;
    }
  },
  
  getWeeklyStats: async (vehicleId: string): Promise<WeeklyStats> => {
    try {
      return await fetchJson<WeeklyStats>(`${API_BASE}/telemetry/stats/weekly/${vehicleId}`);
    } catch (e) {
      console.error('getWeeklyStats error:', e);
      return {
        period: '7_days',
        total_trips: 0,
        total_distance_km: 0,
        total_duration_seconds: 0,
        total_fuel_liters: 0,
        avg_speed_kmh: 0,
        mode_breakdown_seconds: { parked: 0, city: 0, highway: 0, sport: 0, reverse: 0 },
        daily_breakdown: {},
      };
    }
  },
};

// Weekly stats type
export interface WeeklyStats {
  period: string;
  total_trips: number;
  total_distance_km: number;
  total_duration_seconds: number;
  total_fuel_liters: number;
  avg_speed_kmh: number;
  mode_breakdown_seconds: {
    parked: number;
    city: number;
    highway: number;
    sport: number;
    reverse: number;
  };
  daily_breakdown: Record<string, {
    distance_km: number;
    trips: number;
    duration_seconds: number;
  }>;
}

// Health check
export const healthApi = {
  check: () => fetchJson<{ status: string }>('http://localhost:8000/health'),
};