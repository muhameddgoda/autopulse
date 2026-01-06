import { Vehicle } from '../types';

const API_BASE = 'http://localhost:8000/api';

export const vehicleApi = {
  getAll: async (): Promise<Vehicle[]> => {
    const response = await fetch(`${API_BASE}/telemetry/vehicles`);
    return response.json();
  },
};
