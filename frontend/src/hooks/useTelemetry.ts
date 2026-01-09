import { useState, useEffect, useCallback, useRef } from 'react';
import { TelemetryReading, WebSocketMessage } from '../types';

interface UseTelemetryOptions {
  vehicleId: string | null;
  enabled?: boolean;
}

interface UseTelemetryReturn {
  telemetry: TelemetryReading | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

const WS_URL = 'ws://localhost:8000/api/telemetry/stream';

export function useTelemetry({ vehicleId, enabled = true }: UseTelemetryOptions): UseTelemetryReturn {
  const [telemetry, setTelemetry] = useState<TelemetryReading | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

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
          setTelemetry(message.data as TelemetryReading);
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
      
      if (event.code !== 1000 && enabled) {
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      }
    };
  }, [vehicleId, enabled]);

  useEffect(() => {
    if (enabled && vehicleId) connect();

    return () => {
      if (wsRef.current) wsRef.current.close(1000);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
    };
  }, [vehicleId, enabled, connect]);

  return { telemetry, connectionStatus };
}

export default useTelemetry;