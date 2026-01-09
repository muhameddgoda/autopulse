// Theme constants used across the application

// Accent colors
export const ORANGE = "#f97316";
export const ORANGE_DIM = "#f9731640";

// Dark theme colors
export const DARK_BG = "#0a0a0a";
export const DARK_CARD = "#141414";
export const DARK_BORDER = "#262626";
export const DARK_TEXT = "#ffffff";
export const DARK_TEXT_MUTED = "#737373";

// Status colors
export const STATUS_COLORS = {
  success: "#22c55e",
  warning: "#f59e0b",
  error: "#ef4444",
  info: "#3b82f6",
};

// Thresholds for alerts
export const THRESHOLDS = {
  tire_pressure: {
    critical: 28,
    low: 30,
    high: 38,
  },
  engine_temp: {
    critical: 110,
    warning: 100,
  },
  oil_pressure: {
    critical: 15,
    warning: 25,
  },
  battery_voltage: {
    low: 11.5,
    high: 14.5,
  },
};
