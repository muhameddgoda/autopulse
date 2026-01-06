import { useState, useEffect, useCallback, useRef } from 'react';
import { TelemetryReading, WebSocketMessage } from '../types';

interface UseTelemetryOptions {
  vehicleId: string | null;
  enabled?: boolean;
}

interface UseTelemetryReturn {
  telemetry: TelemetryReading | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  error: string | null;
  reconnect: () => void;
}

const WS_URL = 'ws://localhost:8000/api/telemetry/stream';

export function useTelemetry({ vehicleId, enabled = true }: UseTelemetryOptions): UseTelemetryReturn {
  const [telemetry, setTelemetry] = useState<TelemetryReading | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!vehicleId || !enabled) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    setConnectionStatus('connecting');
    setError(null);

    const ws = new WebSocket(`${WS_URL}/${vehicleId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('🔌 WebSocket connected');
      setConnectionStatus('connected');
      setError(null);

      // Start ping interval to keep connection alive
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        
        switch (message.type) {
          case 'telemetry_update':
            if (message.data) {
              setTelemetry(message.data as TelemetryReading);
            }
            break;
          case 'connected':
            console.log('✅ Subscribed to vehicle:', message.vehicle_id);
            break;
          case 'keepalive':
          case 'pong':
            // Connection alive, do nothing
            break;
          default:
            console.log('📨 Message:', message);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onerror = (event) => {
      console.error('❌ WebSocket error:', event);
      setError('Connection error');
      setConnectionStatus('error');
    };

    ws.onclose = (event) => {
      console.log('🔌 WebSocket closed:', event.code, event.reason);
      setConnectionStatus('disconnected');

      // Clear ping interval
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }

      // Auto-reconnect after 3 seconds (unless intentionally closed)
      if (event.code !== 1000 && enabled) {
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('🔄 Attempting to reconnect...');
          connect();
        }, 3000);
      }
    };
  }, [vehicleId, enabled]);

  const reconnect = useCallback(() => {
    connect();
  }, [connect]);

  // Connect when vehicleId changes or enabled changes
  useEffect(() => {
    if (enabled && vehicleId) {
      connect();
    }

    return () => {
      // Cleanup on unmount
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };
  }, [vehicleId, enabled, connect]);

  return {
    telemetry,
    connectionStatus,
    error,
    reconnect,
  };
}
