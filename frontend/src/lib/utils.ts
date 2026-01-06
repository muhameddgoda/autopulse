import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Merge Tailwind classes
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Format number with fixed decimals
export function formatNumber(value: number, decimals: number = 1): string {
  return value.toFixed(decimals);
}

// Format speed for display
export function formatSpeed(kmh: number): string {
  return Math.round(kmh).toString();
}

// Format RPM for display
export function formatRPM(rpm: number): string {
  return Math.round(rpm).toLocaleString();
}

// Format temperature
export function formatTemp(celsius: number): string {
  return `${Math.round(celsius)}°C`;
}

// Format pressure
export function formatPressure(bar: number): string {
  return `${bar.toFixed(1)} bar`;
}

// Format voltage
export function formatVoltage(volts: number): string {
  return `${volts.toFixed(1)}V`;
}

// Format percentage
export function formatPercent(value: number): string {
  return `${Math.round(value)}%`;
}

// Get color based on status
export function getStatusColor(status: 'normal' | 'warning' | 'critical'): string {
  switch (status) {
    case 'critical':
      return 'text-red-500';
    case 'warning':
      return 'text-yellow-500';
    default:
      return 'text-green-500';
  }
}

// Get background color based on status
export function getStatusBgColor(status: 'normal' | 'warning' | 'critical'): string {
  switch (status) {
    case 'critical':
      return 'bg-red-500/20';
    case 'warning':
      return 'bg-yellow-500/20';
    default:
      return 'bg-green-500/20';
  }
}

// Calculate percentage for progress bars
export function calculatePercentage(value: number, min: number, max: number): number {
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}

// Format duration in seconds to human readable
export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

// Format distance
export function formatDistance(km: number): string {
  if (km < 1) {
    return `${Math.round(km * 1000)}m`;
  }
  return `${km.toFixed(1)}km`;
}
