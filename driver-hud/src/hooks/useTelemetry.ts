import { useState, useEffect, useCallback, useRef } from 'react';
import { TelemetryReading, WebSocketMessage, SensorHistory, SensorDataPoint } from '../types';

interface UseTelemetryOptions {
  vehicleId: string | null;
  enabled?: boolean;
  historyLength?: number; // How many data points to keep
}

interface UseTelemetryReturn {
  telemetry: TelemetryReading | null;
  sensorHistory: SensorHistory;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

const WS_URL = 'ws://localhost:8000/api/telemetry/stream';
const MAX_HISTORY_POINTS = 60; // 60 seconds of data at 1Hz

const createEmptyHistory = (): SensorHistory => ({
  engine_temp: [],
  oil_temp: [],
  oil_pressure: [],
  battery_voltage: [],
  fuel_level: [],
});

export function useTelemetry({ 
  vehicleId, 
  enabled = true,
  historyLength = MAX_HISTORY_POINTS 
}: UseTelemetryOptions): UseTelemetryReturn {
  const [telemetry, setTelemetry] = useState<TelemetryReading | null>(null);
  const [sensorHistory, setSensorHistory] = useState<SensorHistory>(createEmptyHistory());
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Add data point to history
  const addToHistory = useCallback((reading: TelemetryReading) => {
    const timestamp = Date.now();
    
    setSensorHistory(prev => {
      const addPoint = (arr: SensorDataPoint[], value: number): SensorDataPoint[] => {
        const newArr = [...arr, { time: timestamp, value }];
        // Keep only the last N points
        return newArr.slice(-historyLength);
      };

      return {
        engine_temp: addPoint(prev.engine_temp, reading.engine_temp),
        oil_temp: addPoint(prev.oil_temp, reading.oil_temp),
        oil_pressure: addPoint(prev.oil_pressure, reading.oil_pressure),
        battery_voltage: addPoint(prev.battery_voltage, reading.battery_voltage),
        fuel_level: addPoint(prev.fuel_level, reading.fuel_level),
      };
    });
  }, [historyLength]);

  const connect = useCallback(() => {
    if (!vehicleId || !enabled) return;

    if (wsRef.current) {
      wsRef.current.close();
    }

    setConnectionStatus('connecting');

    const ws = new WebSocket(`${WS_URL}/${vehicleId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      // Keep connection alive with ping
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        if (message.type === 'telemetry_update' && message.data) {
          const reading = message.data as TelemetryReading;
          setTelemetry(reading);
          addToHistory(reading);
        }
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    };

    ws.onerror = () => {
      setConnectionStatus('error');
    };

    ws.onclose = (event) => {
      setConnectionStatus('disconnected');
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      
      // Auto-reconnect unless intentionally closed
      if (event.code !== 1000 && enabled) {
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      }
    };
  }, [vehicleId, enabled, addToHistory]);

  useEffect(() => {
    if (enabled && vehicleId) {
      connect();
    }

    return () => {
      if (wsRef.current) wsRef.current.close(1000);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
    };
  }, [vehicleId, enabled, connect]);

  return { telemetry, sensorHistory, connectionStatus };
}
