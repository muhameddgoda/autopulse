import { useState, useEffect } from 'react';
import { AlertTriangle, X, Thermometer, Droplet, Fuel, Battery, Gauge } from 'lucide-react';
import { TelemetryReading, THRESHOLDS } from '../types';

const ORANGE = '#f97316';
const DARK_CARD = '#141414';
const DARK_BORDER = '#262626';
const DARK_TEXT = '#ffffff';

export interface Alert {
  id: string;
  type: 'warning' | 'critical';
  icon: React.ReactNode;
  title: string;
  message: string;
  timestamp: Date;
}

// Check telemetry for alerts
export function checkAlerts(telemetry: TelemetryReading): Alert[] {
  const alerts: Alert[] = [];
  const now = new Date();

  // Engine temp
  if (telemetry.engine_temp >= THRESHOLDS.engine_temp.critical) {
    alerts.push({
      id: 'engine_temp_critical',
      type: 'critical',
      icon: <Thermometer className="w-6 h-6" />,
      title: 'Engine Overheating!',
      message: `Engine temperature is critically high at ${telemetry.engine_temp.toFixed(0)}°C. Stop immediately!`,
      timestamp: now,
    });
  } else if (telemetry.engine_temp >= THRESHOLDS.engine_temp.warning) {
    alerts.push({
      id: 'engine_temp_warning',
      type: 'warning',
      icon: <Thermometer className="w-6 h-6" />,
      title: 'Engine Temperature High',
      message: `Engine temperature is ${telemetry.engine_temp.toFixed(0)}°C. Consider slowing down.`,
      timestamp: now,
    });
  }

  // Oil temp
  if (telemetry.oil_temp >= THRESHOLDS.oil_temp.critical) {
    alerts.push({
      id: 'oil_temp_critical',
      type: 'critical',
      icon: <Droplet className="w-6 h-6" />,
      title: 'Oil Temperature Critical!',
      message: `Oil temperature is ${telemetry.oil_temp.toFixed(0)}°C. Stop to prevent engine damage!`,
      timestamp: now,
    });
  } else if (telemetry.oil_temp >= THRESHOLDS.oil_temp.warning) {
    alerts.push({
      id: 'oil_temp_warning',
      type: 'warning',
      icon: <Droplet className="w-6 h-6" />,
      title: 'Oil Temperature High',
      message: `Oil temperature is ${telemetry.oil_temp.toFixed(0)}°C.`,
      timestamp: now,
    });
  }

  // Fuel level
  if (telemetry.fuel_level <= THRESHOLDS.fuel_level.critical) {
    alerts.push({
      id: 'fuel_critical',
      type: 'critical',
      icon: <Fuel className="w-6 h-6" />,
      title: 'Fuel Critically Low!',
      message: `Only ${telemetry.fuel_level.toFixed(0)}% fuel remaining. Find a gas station immediately!`,
      timestamp: now,
    });
  } else if (telemetry.fuel_level <= THRESHOLDS.fuel_level.warning) {
    alerts.push({
      id: 'fuel_warning',
      type: 'warning',
      icon: <Fuel className="w-6 h-6" />,
      title: 'Low Fuel',
      message: `${telemetry.fuel_level.toFixed(0)}% fuel remaining. Consider refueling soon.`,
      timestamp: now,
    });
  }

  // Battery
  if (telemetry.battery_voltage <= THRESHOLDS.battery_voltage.critical) {
    alerts.push({
      id: 'battery_critical',
      type: 'critical',
      icon: <Battery className="w-6 h-6" />,
      title: 'Battery Critical!',
      message: `Battery voltage is ${telemetry.battery_voltage.toFixed(1)}V. Charging system failure!`,
      timestamp: now,
    });
  } else if (telemetry.battery_voltage <= THRESHOLDS.battery_voltage.low) {
    alerts.push({
      id: 'battery_warning',
      type: 'warning',
      icon: <Battery className="w-6 h-6" />,
      title: 'Low Battery Voltage',
      message: `Battery voltage is ${telemetry.battery_voltage.toFixed(1)}V.`,
      timestamp: now,
    });
  }

  // RPM
  if (telemetry.rpm >= THRESHOLDS.rpm.redline) {
    alerts.push({
      id: 'rpm_redline',
      type: 'critical',
      icon: <Gauge className="w-6 h-6" />,
      title: 'RPM at Redline!',
      message: `Engine at ${telemetry.rpm} RPM. Shift up to prevent damage!`,
      timestamp: now,
    });
  } else if (telemetry.rpm >= THRESHOLDS.rpm.warning) {
    alerts.push({
      id: 'rpm_warning',
      type: 'warning',
      icon: <Gauge className="w-6 h-6" />,
      title: 'High RPM',
      message: `Engine at ${telemetry.rpm} RPM. Consider shifting.`,
      timestamp: now,
    });
  }

  // Tire pressure
  const tirePressures = [
    { name: 'Front Left', value: telemetry.tire_pressure_fl },
    { name: 'Front Right', value: telemetry.tire_pressure_fr },
    { name: 'Rear Left', value: telemetry.tire_pressure_rl },
    { name: 'Rear Right', value: telemetry.tire_pressure_rr },
  ];

  for (const tire of tirePressures) {
    if (tire.value && tire.value <= THRESHOLDS.tire_pressure.critical) {
      alerts.push({
        id: `tire_${tire.name.toLowerCase().replace(' ', '_')}_critical`,
        type: 'critical',
        icon: <AlertTriangle className="w-6 h-6" />,
        title: `${tire.name} Tire Pressure Critical!`,
        message: `Pressure is ${tire.value.toFixed(1)} PSI. Stop and check tire!`,
        timestamp: now,
      });
    } else if (tire.value && tire.value <= THRESHOLDS.tire_pressure.low) {
      alerts.push({
        id: `tire_${tire.name.toLowerCase().replace(' ', '_')}_warning`,
        type: 'warning',
        icon: <AlertTriangle className="w-6 h-6" />,
        title: `${tire.name} Tire Pressure Low`,
        message: `Pressure is ${tire.value.toFixed(1)} PSI. Check tire soon.`,
        timestamp: now,
      });
    }
  }

  return alerts;
}

// Alert Overlay Component
interface AlertOverlayProps {
  alert: Alert | null;
  onDismiss: () => void;
}

export function AlertOverlay({ alert, onDismiss }: AlertOverlayProps) {
  if (!alert) return null;

  const bgColor = alert.type === 'critical' ? '#dc2626' : '#f59e0b';
  const borderColor = alert.type === 'critical' ? '#ef4444' : '#fbbf24';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div 
        className="max-w-md w-full mx-4 rounded-2xl p-6 shadow-2xl animate-pulse"
        style={{ 
          backgroundColor: DARK_CARD, 
          border: `2px solid ${borderColor}`,
          boxShadow: `0 0 40px ${bgColor}40`
        }}
      >
        {/* Icon */}
        <div 
          className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center"
          style={{ backgroundColor: `${bgColor}30`, color: bgColor }}
        >
          {alert.icon}
        </div>

        {/* Title */}
        <h2 
          className="text-2xl font-bold text-center mb-2"
          style={{ color: bgColor }}
        >
          {alert.title}
        </h2>

        {/* Message */}
        <p className="text-center mb-6" style={{ color: DARK_TEXT }}>
          {alert.message}
        </p>

        {/* OK Button */}
        <button
          onClick={onDismiss}
          className="w-full py-3 rounded-xl text-white font-semibold transition-all hover:opacity-90"
          style={{ backgroundColor: bgColor }}
        >
          OK
        </button>
      </div>
    </div>
  );
}

// Hook to manage alerts
export function useAlerts(telemetry: TelemetryReading | null) {
  const [currentAlert, setCurrentAlert] = useState<Alert | null>(null);
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!telemetry) return;

    const alerts = checkAlerts(telemetry);
    
    // Find first undismissed critical alert, then warning
    const criticalAlert = alerts.find(a => a.type === 'critical' && !dismissedAlerts.has(a.id));
    const warningAlert = alerts.find(a => a.type === 'warning' && !dismissedAlerts.has(a.id));
    
    const nextAlert = criticalAlert || warningAlert;
    
    if (nextAlert && (!currentAlert || currentAlert.id !== nextAlert.id)) {
      setCurrentAlert(nextAlert);
    }
  }, [telemetry, dismissedAlerts]);

  const dismissAlert = () => {
    if (currentAlert) {
      setDismissedAlerts(prev => new Set([...prev, currentAlert.id]));
      setCurrentAlert(null);
      
      // Clear dismissed alerts after 30 seconds (allow re-triggering)
      setTimeout(() => {
        setDismissedAlerts(prev => {
          const next = new Set(prev);
          next.delete(currentAlert.id);
          return next;
        });
      }, 30000);
    }
  };

  return { currentAlert, dismissAlert };
}
