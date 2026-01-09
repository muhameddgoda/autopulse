import React from 'react';
import { AlertTriangle, Fuel, Thermometer, Gauge, X } from 'lucide-react';

// Theme colors
const RED = '#ef4444';
const YELLOW = '#eab308';

interface Alert {
  type: 'fuel' | 'rpm' | 'temperature' | 'oil';
  message: string;
  severity: 'warning' | 'critical';
}

interface AlertOverlayProps {
  alerts: Alert[];
  onDismiss?: (index: number) => void;
}

export const AlertOverlay: React.FC<AlertOverlayProps> = ({ alerts, onDismiss }) => {
  if (!alerts || alerts.length === 0) return null;

  const getIcon = (type: string) => {
    switch (type) {
      case 'fuel':
        return <Fuel size={24} />;
      case 'rpm':
        return <Gauge size={24} />;
      case 'temperature':
        return <Thermometer size={24} />;
      default:
        return <AlertTriangle size={24} />;
    }
  };

  const getColor = (severity: string) => {
    return severity === 'critical' ? RED : YELLOW;
  };

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-md">
      {alerts.map((alert, index) => (
        <div
          key={`${alert.type}-${index}`}
          className="flex items-center gap-3 p-4 rounded-lg shadow-lg"
          style={{
            backgroundColor: `${getColor(alert.severity)}20`,
            borderLeft: `4px solid ${getColor(alert.severity)}`,
            animation: alert.severity === 'critical' ? 'pulse 1s infinite' : 'none',
          }}
        >
          <div style={{ color: getColor(alert.severity) }}>
            {getIcon(alert.type)}
          </div>
          <div className="flex-1">
            <div 
              className="font-bold uppercase text-sm"
              style={{ color: getColor(alert.severity) }}
            >
              {alert.type} {alert.severity}
            </div>
            <div className="text-white text-sm">{alert.message}</div>
          </div>
          {onDismiss && (
            <button
              onClick={() => onDismiss(index)}
              className="text-gray-400 hover:text-white transition-colors p-1"
              aria-label="Dismiss alert"
            >
              <X size={18} />
            </button>
          )}
        </div>
      ))}
    </div>
  );
};

// Support both named and default imports
export default AlertOverlay;