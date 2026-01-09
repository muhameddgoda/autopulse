import { useEffect, useMemo, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { vehicleApi } from "../lib/api";
import { Vehicle } from "../types";

// ============================================
// THEME CONSTANTS
// ============================================
const THEME = {
  bg: "#0a0a0a",
  card: "#111111",
  cardHover: "#161616",
  border: "#1e1e1e",
  borderLight: "#2a2a2a",
  text: "#ffffff",
  textMuted: "#737373",
  textDim: "#525252",
  orange: "#f97316",
  orangeGlow: "rgba(249, 115, 22, 0.15)",
  green: "#22c55e",
  greenGlow: "rgba(34, 197, 94, 0.15)",
  blue: "#3b82f6",
  blueGlow: "rgba(59, 130, 246, 0.15)",
  yellow: "#eab308",
  yellowGlow: "rgba(234, 179, 8, 0.15)",
  red: "#ef4444",
  redGlow: "rgba(239, 68, 68, 0.15)",
};

const BEHAVIOR_CONFIG: Record<
  string,
  { color: string; glow: string; label: string }
> = {
  exemplary: {
    color: "#22c55e",
    glow: "rgba(34, 197, 94, 0.2)",
    label: "Exemplary",
  },
  calm: { color: "#3b82f6", glow: "rgba(59, 130, 246, 0.2)", label: "Calm" },
  normal: { color: "#eab308", glow: "rgba(234, 179, 8, 0.2)", label: "Normal" },
  aggressive: {
    color: "#f97316",
    glow: "rgba(249, 115, 22, 0.2)",
    label: "Aggressive",
  },
  dangerous: {
    color: "#ef4444",
    glow: "rgba(239, 68, 68, 0.2)",
    label: "Dangerous",
  },
};

type Scope = "trip" | "7d" | "30d";

// ============================================
// API TYPES
// ============================================
interface TripWithScore {
  id: string;
  vehicle_id: string;
  start_time: string;
  end_time: string | null;
  is_active: boolean;
  distance_km: number | null;
  duration_seconds: number | null;
  avg_speed_kmh: number | null;
  max_speed_kmh: number | null;
  driver_score: number | null;
  behavior_label: string | null;
  risk_level: string | null;
  harsh_brake_count: number | null;
  harsh_accel_count: number | null;
  speeding_percentage: number | null;
  ml_enhanced: boolean | null;
}

interface TripScoreDetails {
  trip_id: string;
  driver_score: number | null;
  behavior_label: string | null;
  risk_level: string | null;
  harsh_brake_count: number;
  harsh_accel_count: number;
  speeding_percentage: number;
  avg_speed_kmh: number | null;
  max_speed_kmh: number | null;
  duration_seconds: number | null;
  distance_km: number | null;
  ml_enhanced: boolean | null;
  summary: {
    duration_minutes: number;
    distance_km: number;
    avg_speed_kmh: number;
    max_speed_kmh: number;
    harsh_events_total: number;
    speeding_percentage: number;
  };
  insights: string[];
  recommendations: string[];
}

interface DriverSummary {
  total_trips: number;
  score_statistics: { average: number; minimum: number; maximum: number };
  overall_behavior: string;
  trend: string;
  totals: {
    distance_km: number;
    duration_hours: number;
    harsh_brakes: number;
    harsh_accelerations: number;
  };
  events_per_100km: { harsh_brakes: number; harsh_accels: number };
}

interface ScoreHistoryItem {
  trip_id: string;
  date: string;
  score: number;
  behavior: string;
  distance_km: number | null;
}

interface TripTimelinePoint {
  time_offset: number;
  time: string;
  speed_kmh: number;
  rpm: number;
  score: number;
  is_harsh_brake: boolean;
  is_harsh_accel: boolean;
  is_speeding: boolean;
  acceleration_g: number;
}

// ============================================
// API FUNCTIONS - Using /api/scoring/* endpoints
// ============================================
const API_BASE = "http://localhost:8000/api/scoring";
const ML_API_BASE = "http://localhost:8000/api/telemetry/ml";

interface MLStatus {
  ml_available: boolean;
  behavior_model: boolean;
  anomaly_model: boolean;
}

async function getMLStatus(vehicleId: string): Promise<MLStatus | null> {
  try {
    const res = await fetch(
      `${ML_API_BASE}/models/status?vehicle_id=${vehicleId}`
    );
    if (!res.ok) return null;
    const data = await res.json();

    // Parse the response to determine if models exist for this vehicle
    const models = data.models || {};
    const hasBehavior = Object.keys(models).some(
      (k) =>
        k.includes("behavior") &&
        k.includes(vehicleId.substring(0, 8)) &&
        models[k]?.exists
    );
    const hasAnomaly = Object.keys(models).some(
      (k) =>
        k.includes("anomaly") &&
        k.includes(vehicleId.substring(0, 8)) &&
        models[k]?.exists
    );

    return {
      ml_available: data.ml_available ?? false,
      behavior_model: hasBehavior,
      anomaly_model: hasAnomaly,
    };
  } catch (e) {
    console.error("Failed to fetch ML status:", e);
    return null;
  }
}

async function getTripsWithScores(
  vehicleId: string,
  limit = 50
): Promise<TripWithScore[]> {
  try {
    const res = await fetch(
      `${API_BASE}/trips/${vehicleId}?limit=${limit}&include_active=false`
    );
    if (!res.ok) return [];
    return res.json();
  } catch (e) {
    console.error("Failed to fetch trips:", e);
    return [];
  }
}

async function getTripScoreDetails(
  tripId: string
): Promise<TripScoreDetails | null> {
  try {
    const res = await fetch(`${API_BASE}/trips/${tripId}/details`);
    if (!res.ok) return null;
    return res.json();
  } catch (e) {
    console.error("Failed to fetch trip details:", e);
    return null;
  }
}

async function getDriverSummary(
  vehicleId: string,
  days: number
): Promise<DriverSummary | null> {
  try {
    const res = await fetch(`${API_BASE}/summary/${vehicleId}?days=${days}`);
    if (!res.ok) return null;
    return res.json();
  } catch (e) {
    console.error("Failed to fetch summary:", e);
    return null;
  }
}

async function getScoreHistory(
  vehicleId: string,
  limit = 10,
  days?: number
): Promise<ScoreHistoryItem[]> {
  try {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (days) params.append("days", days.toString());
    const res = await fetch(`${API_BASE}/history/${vehicleId}?${params}`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.history || [];
  } catch (e) {
    console.error("Failed to fetch history:", e);
    return [];
  }
}

async function getTripTimeline(tripId: string): Promise<TripTimelinePoint[]> {
  try {
    const res = await fetch(`${API_BASE}/trips/${tripId}/timeline`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.timeline || [];
  } catch (e) {
    console.error("Failed to fetch trip timeline:", e);
    return [];
  }
}

// ============================================
// COMPONENTS
// ============================================

function ScoreGauge({ score, size = 160 }: { score: number; size?: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return THEME.green;
    if (s >= 60) return THEME.yellow;
    if (s >= 40) return THEME.orange;
    return THEME.red;
  };

  const color = getColor(score);
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75;
  const filledLength = (score / 100) * arcLength;
  const center = size / 2;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-full">
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={THEME.border}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${arcLength} ${circumference}`}
          transform={`rotate(135 ${center} ${center})`}
        />
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${filledLength} ${circumference}`}
          transform={`rotate(135 ${center} ${center})`}
          style={{
            transition: "stroke-dasharray 0.8s ease-out",
            filter: `drop-shadow(0 0 8px ${color}40)`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-5xl font-bold tracking-tight" style={{ color }}>
          {score.toFixed(0)}
        </span>
        <span className="text-sm mt-1" style={{ color: THEME.textMuted }}>
          out of 100
        </span>
      </div>
    </div>
  );
}

function BehaviorRing({
  behavior,
  score,
}: {
  behavior: string;
  score: number;
}) {
  const config = BEHAVIOR_CONFIG[behavior] || BEHAVIOR_CONFIG.normal;
  const percentage = score;

  const data = [
    { name: "Score", value: percentage },
    { name: "Remaining", value: 100 - percentage },
  ];

  return (
    <div className="relative w-32 h-32">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={38}
            outerRadius={50}
            startAngle={90}
            endAngle={-270}
            paddingAngle={0}
            dataKey="value"
          >
            <Cell fill={config.color} />
            <Cell fill={THEME.border} />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xs" style={{ color: config.color }}>
          ●
        </span>
        <span className="text-xs mt-1" style={{ color: config.color }}>
          {config.label}
        </span>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  subValue,
  trend,
}: {
  icon: string;
  label: string;
  value: string | number;
  subValue?: string;
  trend?: "up" | "down" | "stable";
}) {
  return (
    <div
      className="rounded-xl p-4 transition-all duration-200 hover:scale-[1.02]"
      style={{
        backgroundColor: THEME.card,
        border: `1px solid ${THEME.border}`,
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{icon}</span>
        <span
          className="text-xs uppercase tracking-wider"
          style={{ color: THEME.textMuted }}
        >
          {label}
        </span>
        {trend && (
          <span className="ml-auto text-xs">
            {trend === "up" && <span style={{ color: THEME.green }}>↑</span>}
            {trend === "down" && <span style={{ color: THEME.red }}>↓</span>}
            {trend === "stable" && (
              <span style={{ color: THEME.textMuted }}>→</span>
            )}
          </span>
        )}
      </div>
      <div className="text-2xl font-bold" style={{ color: THEME.text }}>
        {value}
      </div>
      {subValue && (
        <div className="text-xs mt-1" style={{ color: THEME.textMuted }}>
          {subValue}
        </div>
      )}
    </div>
  );
}

function ScoreHistoryChart({ data }: { data: ScoreHistoryItem[] }) {
  const chartData = data.map((item, index) => ({
    name: index + 1,
    score: item.score,
    date: new Date(item.date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    }),
  }));

  return (
    <div className="h-40">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 5, right: 5, left: -20, bottom: 5 }}
        >
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={THEME.orange} stopOpacity={0.3} />
              <stop offset="95%" stopColor={THEME.orange} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="name"
            tick={{ fontSize: 10, fill: THEME.textMuted }}
            axisLine={{ stroke: THEME.border }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: THEME.textMuted }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: THEME.card,
              border: `1px solid ${THEME.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number) => [`${value.toFixed(0)}`, "Score"]}
            labelFormatter={(_, payload) => payload[0]?.payload?.date || ""}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke={THEME.orange}
            strokeWidth={2}
            fill="url(#scoreGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function TripTimelineChart({ data }: { data: TripTimelinePoint[] }) {
  // Sample data if there are too many points (for performance)
  const sampledData =
    data.length > 60
      ? data.filter((_, i) => i % Math.ceil(data.length / 60) === 0)
      : data;

  const chartData = sampledData.map((point) => ({
    time: point.time_offset,
    score: point.score,
    speed: point.speed_kmh,
    label: `${Math.floor(point.time_offset / 60)}:${(point.time_offset % 60)
      .toString()
      .padStart(2, "0")}`,
    hasEvent: point.is_harsh_brake || point.is_harsh_accel,
    eventType: point.is_harsh_brake
      ? "brake"
      : point.is_harsh_accel
      ? "accel"
      : null,
  }));

  return (
    <div className="h-40">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 5, right: 5, left: -20, bottom: 5 }}
        >
          <defs>
            <linearGradient id="tripScoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={THEME.green} stopOpacity={0.3} />
              <stop offset="95%" stopColor={THEME.green} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={THEME.blue} stopOpacity={0.2} />
              <stop offset="95%" stopColor={THEME.blue} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9, fill: THEME.textMuted }}
            axisLine={{ stroke: THEME.border }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="score"
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: THEME.textMuted }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            yAxisId="speed"
            orientation="right"
            domain={[0, "dataMax"]}
            tick={{ fontSize: 10, fill: THEME.textMuted }}
            axisLine={false}
            tickLine={false}
            hide
          />
          <Tooltip
            contentStyle={{
              backgroundColor: THEME.card,
              border: `1px solid ${THEME.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number, name: string) => {
              if (name === "score") return [`${value.toFixed(0)}`, "Score"];
              if (name === "speed")
                return [`${value.toFixed(0)} km/h`, "Speed"];
              return [value, name];
            }}
            labelFormatter={(_, payload) => {
              const point = payload[0]?.payload;
              if (!point) return "";
              let label = `Time: ${point.label}`;
              if (point.eventType === "brake") label += " ⚠️ Harsh Brake";
              if (point.eventType === "accel") label += " ⚡ Hard Accel";
              return label;
            }}
          />
          <Area
            yAxisId="speed"
            type="monotone"
            dataKey="speed"
            stroke={THEME.blue}
            strokeWidth={1}
            fill="url(#speedGradient)"
            strokeOpacity={0.5}
          />
          <Area
            yAxisId="score"
            type="monotone"
            dataKey="score"
            stroke={THEME.green}
            strokeWidth={2}
            fill="url(#tripScoreGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function HarshEventsCard({
  brakes,
  accels,
}: {
  brakes: number;
  accels: number;
}) {
  return (
    <div
      className="rounded-xl p-4"
      style={{
        backgroundColor: THEME.card,
        border: `1px solid ${THEME.border}`,
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">⚠️</span>
        <span
          className="text-xs uppercase tracking-wider"
          style={{ color: THEME.textMuted }}
        >
          Harsh Events
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-3xl font-bold" style={{ color: THEME.text }}>
            {brakes}
          </div>
          <div className="text-xs" style={{ color: THEME.textMuted }}>
            Hard Brakes
          </div>
        </div>
        <div>
          <div className="text-3xl font-bold" style={{ color: THEME.text }}>
            {accels}
          </div>
          <div className="text-xs" style={{ color: THEME.textMuted }}>
            Hard Accels
          </div>
        </div>
      </div>
    </div>
  );
}

function Per100KMCard({ brakes, accels }: { brakes: number; accels: number }) {
  return (
    <div
      className="rounded-xl p-4"
      style={{
        backgroundColor: THEME.card,
        border: `1px solid ${THEME.border}`,
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">📊</span>
        <span
          className="text-xs uppercase tracking-wider"
          style={{ color: THEME.textMuted }}
        >
          Per 100KM
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-3xl font-bold" style={{ color: THEME.text }}>
            {brakes.toFixed(1)}
          </div>
          <div className="text-xs" style={{ color: THEME.textMuted }}>
            Brake Events
          </div>
        </div>
        <div>
          <div className="text-3xl font-bold" style={{ color: THEME.text }}>
            {accels.toFixed(1)}
          </div>
          <div className="text-xs" style={{ color: THEME.textMuted }}>
            Accel Events
          </div>
        </div>
      </div>
    </div>
  );
}

function AnomalyCard({ hasAnomalies }: { hasAnomalies: boolean }) {
  return (
    <div
      className="rounded-xl p-4"
      style={{
        backgroundColor: THEME.card,
        border: `1px solid ${THEME.border}`,
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg">🔍</span>
        <span
          className="text-xs uppercase tracking-wider"
          style={{ color: THEME.textMuted }}
        >
          Anomaly Detection
        </span>
      </div>

      <div
        className="flex items-center gap-2 p-3 rounded-lg"
        style={{
          backgroundColor: hasAnomalies ? THEME.redGlow : THEME.greenGlow,
        }}
      >
        <span style={{ color: hasAnomalies ? THEME.red : THEME.green }}>
          {hasAnomalies ? "⚠️" : "✓"}
        </span>
        <div>
          <div
            className="text-sm font-medium"
            style={{ color: hasAnomalies ? THEME.red : THEME.green }}
          >
            {hasAnomalies ? "Anomalies Detected" : "All Clear"}
          </div>
          <div className="text-xs" style={{ color: THEME.textMuted }}>
            {hasAnomalies ? "Review driving patterns" : "No anomalies detected"}
          </div>
        </div>
      </div>
    </div>
  );
}

function TripListItem({
  trip,
  isSelected,
  onClick,
}: {
  trip: TripWithScore;
  isSelected: boolean;
  onClick: () => void;
}) {
  const score = trip.driver_score;
  const getScoreColor = (s: number | null) => {
    if (s === null) return THEME.textMuted;
    if (s >= 80) return THEME.green;
    if (s >= 60) return THEME.yellow;
    if (s >= 40) return THEME.orange;
    return THEME.red;
  };

  const date = new Date(trip.start_time);
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  const formattedTime = date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });

  return (
    <div
      onClick={onClick}
      className="flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all duration-150"
      style={{
        backgroundColor: isSelected ? THEME.cardHover : "transparent",
        border: `1px solid ${isSelected ? THEME.borderLight : "transparent"}`,
      }}
    >
      <div
        className="w-8 h-8 rounded-full flex items-center justify-center text-xs"
        style={{
          backgroundColor: `${getScoreColor(score)}15`,
          border: `2px solid ${getScoreColor(score)}`,
        }}
      >
        {trip.behavior_label === "exemplary" && "✓"}
        {trip.behavior_label === "calm" && "●"}
        {trip.behavior_label === "normal" && "○"}
        {trip.behavior_label === "aggressive" && "!"}
        {trip.behavior_label === "dangerous" && "⚠"}
        {!trip.behavior_label && "-"}
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium" style={{ color: THEME.text }}>
          {formattedDate}, {formattedTime}
        </div>
        <div className="text-xs" style={{ color: THEME.textMuted }}>
          {trip.distance_km?.toFixed(1) || "?"} km •{" "}
          {Math.round((trip.duration_seconds || 0) / 60)}m
        </div>
      </div>

      <div className="text-right">
        <div
          className="text-lg font-bold"
          style={{ color: getScoreColor(score) }}
        >
          {score?.toFixed(1) || "--"}
        </div>
        <div
          className="text-[10px] px-1.5 py-0.5 rounded"
          style={{
            backgroundColor: `${getScoreColor(score)}15`,
            color: getScoreColor(score),
          }}
        >
          {score !== null && score >= 60
            ? "Good"
            : score !== null
            ? "Review"
            : "N/A"}
        </div>
      </div>
    </div>
  );
}

function HealthIndicator({
  label,
  value,
  status,
}: {
  label: string;
  value: number;
  status: "good" | "warning" | "critical";
}) {
  const colors = {
    good: THEME.green,
    warning: THEME.yellow,
    critical: THEME.red,
  };

  return (
    <div
      className="flex items-center justify-between p-3 rounded-lg"
      style={{
        backgroundColor: THEME.card,
        border: `1px solid ${THEME.border}`,
      }}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: colors[status] }}
        />
        <span className="text-sm" style={{ color: THEME.text }}>
          {label}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span
          className="text-xs px-2 py-0.5 rounded"
          style={{
            backgroundColor: `${colors[status]}15`,
            color: colors[status],
          }}
        >
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
        <span className="text-sm font-medium" style={{ color: THEME.text }}>
          {value}%
        </span>
      </div>
    </div>
  );
}

// ============================================
// MAIN COMPONENT
// ============================================
export default function AnalyticsPage() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);
  const [scope, setScope] = useState<Scope>("7d");

  const [trips, setTrips] = useState<TripWithScore[]>([]);
  const [selectedTripId, setSelectedTripId] = useState<string | null>(null);
  const [tripDetails, setTripDetails] = useState<TripScoreDetails | null>(null);
  const [tripTimeline, setTripTimeline] = useState<TripTimelinePoint[]>([]);
  const [summary, setSummary] = useState<DriverSummary | null>(null);
  const [scoreHistory, setScoreHistory] = useState<ScoreHistoryItem[]>([]);
  const [mlStatus, setMlStatus] = useState<MLStatus | null>(null);

  const scopeDays = useMemo(
    () => (scope === "7d" ? 7 : scope === "30d" ? 30 : 1),
    [scope]
  );

  const filteredTrips = useMemo(() => {
    if (scope === "trip") return trips;
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - scopeDays);
    return trips.filter((t) => new Date(t.start_time) >= cutoff);
  }, [trips, scope, scopeDays]);

  const displayData = useMemo(() => {
    if (scope === "trip" && tripDetails) {
      return {
        score: tripDetails.driver_score || 0,
        behavior: tripDetails.behavior_label || "normal",
        risk: tripDetails.risk_level || "medium",
        harshBrakes: tripDetails.harsh_brake_count,
        harshAccels: tripDetails.harsh_accel_count,
        speedingPct: tripDetails.speeding_percentage,
        avgSpeed: tripDetails.avg_speed_kmh || 0,
        maxSpeed: tripDetails.max_speed_kmh || 0,
        distance: tripDetails.distance_km || 0,
        duration: tripDetails.duration_seconds || 0,
        insights: tripDetails.insights || [],
        recommendations: tripDetails.recommendations || [],
        totalTrips: 1,
        per100kmBrakes: tripDetails.distance_km
          ? (tripDetails.harsh_brake_count / tripDetails.distance_km) * 100
          : 0,
        per100kmAccels: tripDetails.distance_km
          ? (tripDetails.harsh_accel_count / tripDetails.distance_km) * 100
          : 0,
      };
    } else if (summary) {
      return {
        score: summary.score_statistics.average,
        behavior: summary.overall_behavior,
        risk:
          summary.score_statistics.average >= 70
            ? "low"
            : summary.score_statistics.average >= 50
            ? "medium"
            : "high",
        harshBrakes: summary.totals.harsh_brakes,
        harshAccels: summary.totals.harsh_accelerations,
        speedingPct: 0,
        avgSpeed: 0,
        maxSpeed: 0,
        distance: summary.totals.distance_km,
        duration: summary.totals.duration_hours * 3600,
        insights: [
          `${summary.total_trips} trips analyzed`,
          `Average score: ${summary.score_statistics.average.toFixed(0)}`,
          `Best score: ${summary.score_statistics.maximum.toFixed(0)}`,
        ],
        recommendations:
          summary.trend === "improving"
            ? ["Great progress! Keep up the good work."]
            : summary.trend === "declining"
            ? ["Focus on smoother braking and acceleration."]
            : ["Maintain your current driving habits."],
        totalTrips: summary.total_trips,
        per100kmBrakes: summary.events_per_100km.harsh_brakes,
        per100kmAccels: summary.events_per_100km.harsh_accels,
      };
    }
    return null;
  }, [scope, tripDetails, summary]);

  const trendDirection = useMemo(() => {
    if (!summary) return "stable";
    // Map backend trend values: improving/declining/stable
    return summary.trend;
  }, [summary]);

  // Load vehicle
  useEffect(() => {
    async function fetchVehicle() {
      try {
        const vehicles = await vehicleApi.getAll();
        if (vehicles.length > 0) setVehicle(vehicles[0]);
      } catch (err) {
        console.error("Failed to fetch vehicle:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchVehicle();
  }, []);

  // Load ML status when vehicle is available
  useEffect(() => {
    if (!vehicle) return;

    async function fetchMLStatus() {
      const status = await getMLStatus(vehicle!.id);
      setMlStatus(status);
    }
    fetchMLStatus();
  }, [vehicle]);

  // Load trips when vehicle is available
  useEffect(() => {
    if (!vehicle) return;

    async function fetchTrips() {
      const tripsData = await getTripsWithScores(vehicle!.id, 100);
      const completedTrips = tripsData.filter((t) => !t.is_active);
      setTrips(completedTrips);

      if (completedTrips.length > 0 && !selectedTripId) {
        setSelectedTripId(completedTrips[0].id);
      }
    }
    fetchTrips();
  }, [vehicle]);

  // Load summary when scope changes (not trip mode)
  useEffect(() => {
    if (!vehicle || scope === "trip") return;

    async function fetchSummary() {
      const days = scope === "7d" ? 7 : 30;
      const data = await getDriverSummary(vehicle!.id, days);
      setSummary(data);
    }
    fetchSummary();
  }, [vehicle, scope]);

  // Load score history based on scope
  useEffect(() => {
    if (!vehicle) return;

    async function fetchHistory() {
      if (scope === "trip") {
        // In trip mode, we don't show score history in the chart
        setScoreHistory([]);
      } else {
        // Load history filtered by the scope's time range
        const days = scope === "7d" ? 7 : 30;
        const history = await getScoreHistory(vehicle!.id, 50, days);
        setScoreHistory(history);
      }
    }
    fetchHistory();
  }, [vehicle, scope]);

  // Load trip details and timeline when selected (trip mode only)
  useEffect(() => {
    if (!selectedTripId || scope !== "trip") {
      setTripDetails(null);
      setTripTimeline([]);
      return;
    }

    async function fetchTripData() {
      const [details, timeline] = await Promise.all([
        getTripScoreDetails(selectedTripId!),
        getTripTimeline(selectedTripId!),
      ]);
      setTripDetails(details);
      setTripTimeline(timeline);
    }
    fetchTripData();
  }, [selectedTripId, scope]);

  if (loading) {
    return (
      <div
        className="h-full flex items-center justify-center"
        style={{ backgroundColor: THEME.bg }}
      >
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-t-orange-500 border-gray-700 rounded-full animate-spin mx-auto mb-4" />
          <div style={{ color: THEME.textMuted }}>Loading analytics...</div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="h-full flex flex-col overflow-hidden"
      style={{ backgroundColor: THEME.bg }}
    >
      {/* Header */}
      <div
        className="flex-shrink-0 px-6 py-4 flex items-center justify-between"
        style={{ borderBottom: `1px solid ${THEME.border}` }}
      >
        <div>
          <h1
            className="text-xl font-bold flex items-center gap-2"
            style={{ color: THEME.text }}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: THEME.orange }}
            />
            Analytics
          </h1>
          <p className="text-xs mt-0.5" style={{ color: THEME.textMuted }}>
            {scope === "trip"
              ? "Single Trip Analysis"
              : `Last ${scopeDays} days`}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {mlStatus?.behavior_model && mlStatus?.anomaly_model ? (
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
              style={{ backgroundColor: THEME.greenGlow, color: THEME.green }}
            >
              <span>✓</span>
              <span>ML Enhanced</span>
            </div>
          ) : (
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs cursor-pointer hover:opacity-80"
              style={{ backgroundColor: THEME.yellowGlow, color: THEME.yellow }}
              title="Train ML models for enhanced scoring"
            >
              <span>⚡</span>
              <span>ML Training Required</span>
            </div>
          )}

          <div
            className="flex rounded-lg overflow-hidden"
            style={{ border: `1px solid ${THEME.border}` }}
          >
            {(["trip", "7d", "30d"] as Scope[]).map((s) => (
              <button
                key={s}
                onClick={() => setScope(s)}
                className="px-4 py-2 text-sm font-medium transition-colors"
                style={{
                  backgroundColor: scope === s ? THEME.orange : "transparent",
                  color: scope === s ? "#000" : THEME.textMuted,
                }}
              >
                {s === "trip" ? "Trip" : s.toUpperCase()}
              </button>
            ))}
          </div>

          <div
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg"
            style={{
              backgroundColor: THEME.card,
              border: `1px solid ${THEME.border}`,
            }}
          >
            <span>🚗</span>
            <span className="text-sm" style={{ color: THEME.text }}>
              {vehicle?.model || "Porsche"} {vehicle?.variant || "911"}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Section */}
        <div className="flex-1 p-6 overflow-y-auto">
          {displayData ? (
            <div className="grid grid-cols-12 gap-4">
              {/* Row 1 */}
              <div className="col-span-3">
                <div
                  className="rounded-xl p-6 h-full flex flex-col items-center justify-center"
                  style={{
                    backgroundColor: THEME.card,
                    border: `1px solid ${THEME.border}`,
                  }}
                >
                  <div
                    className="text-xs uppercase tracking-wider mb-4 flex items-center gap-2"
                    style={{ color: THEME.textMuted }}
                  >
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ backgroundColor: THEME.orange }}
                    />
                    Driver Score
                    {trendDirection !== "stable" && (
                      <span
                        style={{
                          color:
                            trendDirection === "improving"
                              ? THEME.green
                              : THEME.red,
                        }}
                      >
                        {trendDirection === "improving" ? "↑" : "↓"}
                      </span>
                    )}
                  </div>
                  <ScoreGauge score={displayData.score} />
                  <div
                    className="mt-4 px-3 py-1.5 rounded-full text-sm font-medium"
                    style={{
                      backgroundColor:
                        BEHAVIOR_CONFIG[displayData.behavior]?.glow ||
                        THEME.orangeGlow,
                      color:
                        BEHAVIOR_CONFIG[displayData.behavior]?.color ||
                        THEME.orange,
                    }}
                  >
                    ● {BEHAVIOR_CONFIG[displayData.behavior]?.label || "Normal"}
                  </div>
                </div>
              </div>

              <div className="col-span-5">
                <div
                  className="rounded-xl p-4 h-full"
                  style={{
                    backgroundColor: THEME.card,
                    border: `1px solid ${THEME.border}`,
                  }}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-lg">
                      {scope === "trip" ? "🚗" : "📈"}
                    </span>
                    <span
                      className="text-xs uppercase tracking-wider"
                      style={{ color: THEME.textMuted }}
                    >
                      {scope === "trip" ? "Trip Behavior" : "Score History"}
                    </span>
                    <span
                      className="ml-auto text-xs"
                      style={{ color: THEME.textMuted }}
                    >
                      {scope === "trip"
                        ? tripTimeline.length > 0
                          ? `${tripTimeline.length} readings`
                          : "No data"
                        : `${scoreHistory.length} trips in ${
                            scope === "7d" ? "7 days" : "30 days"
                          }`}
                    </span>
                  </div>
                  {scope === "trip" ? (
                    tripTimeline.length > 0 ? (
                      <TripTimelineChart data={tripTimeline} />
                    ) : (
                      <div
                        className="h-40 flex items-center justify-center"
                        style={{ color: THEME.textMuted }}
                      >
                        No telemetry data for this trip
                      </div>
                    )
                  ) : scoreHistory.length > 0 ? (
                    <ScoreHistoryChart data={scoreHistory} />
                  ) : (
                    <div
                      className="h-40 flex items-center justify-center"
                      style={{ color: THEME.textMuted }}
                    >
                      No trips in this period
                    </div>
                  )}
                </div>
              </div>

              <div className="col-span-4">
                <div className="grid grid-cols-2 gap-3 h-full">
                  <StatCard
                    icon="🚗"
                    label="Trips"
                    value={displayData.totalTrips}
                  />
                  <StatCard
                    icon="📍"
                    label="Distance"
                    value={`${displayData.distance.toFixed(1)}`}
                    subValue="km"
                  />
                  <StatCard
                    icon="⏱️"
                    label="Time"
                    value={`${(displayData.duration / 3600).toFixed(1)}`}
                    subValue="hrs"
                  />
                  <StatCard
                    icon="⭐"
                    label="Avg Score"
                    value={displayData.score.toFixed(0)}
                  />
                </div>
              </div>

              {/* Row 2 */}
              <div className="col-span-3">
                <div
                  className="rounded-xl p-4"
                  style={{
                    backgroundColor: THEME.card,
                    border: `1px solid ${THEME.border}`,
                  }}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-lg">🎯</span>
                    <span
                      className="text-xs uppercase tracking-wider"
                      style={{ color: THEME.textMuted }}
                    >
                      Behavior
                    </span>
                  </div>
                  <div className="flex items-center justify-center">
                    <BehaviorRing
                      behavior={displayData.behavior}
                      score={displayData.score}
                    />
                  </div>
                </div>
              </div>

              <div className="col-span-3">
                <HarshEventsCard
                  brakes={displayData.harshBrakes}
                  accels={displayData.harshAccels}
                />
              </div>

              <div className="col-span-3">
                <Per100KMCard
                  brakes={displayData.per100kmBrakes}
                  accels={displayData.per100kmAccels}
                />
              </div>

              <div className="col-span-3">
                <AnomalyCard hasAnomalies={displayData.score < 50} />
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="text-5xl mb-4">📊</div>
                <h2
                  className="text-lg font-semibold mb-2"
                  style={{ color: THEME.text }}
                >
                  No Data Available
                </h2>
                <p className="text-sm" style={{ color: THEME.textMuted }}>
                  {scope === "trip"
                    ? "Select a trip from the sidebar to view details"
                    : "No trips found in this time range"}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar */}
        <div
          className="w-80 flex-shrink-0 flex flex-col overflow-hidden"
          style={{ borderLeft: `1px solid ${THEME.border}` }}
        >
          <div
            className="flex-shrink-0 p-4"
            style={{ borderBottom: `1px solid ${THEME.border}` }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-lg">📋</span>
                <span
                  className="text-sm font-medium"
                  style={{ color: THEME.text }}
                >
                  {scope === "trip" ? "Select Trip" : "Recent Trips"}
                </span>
              </div>
              <span
                className="text-xs px-2 py-0.5 rounded"
                style={{ backgroundColor: THEME.card, color: THEME.textMuted }}
              >
                {filteredTrips.length}
              </span>
            </div>
            <p className="text-xs mt-1" style={{ color: THEME.textMuted }}>
              {scope === "trip"
                ? "Click to analyze"
                : `Showing last ${scopeDays} days`}
            </p>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {filteredTrips.length > 0 ? (
              filteredTrips.map((trip) => (
                <TripListItem
                  key={trip.id}
                  trip={trip}
                  isSelected={selectedTripId === trip.id && scope === "trip"}
                  onClick={() => {
                    setSelectedTripId(trip.id);
                    setScope("trip");
                  }}
                />
              ))
            ) : (
              <div className="text-center py-8">
                <div className="text-3xl mb-2">🚫</div>
                <div className="text-sm" style={{ color: THEME.textMuted }}>
                  No trips in this period
                </div>
              </div>
            )}
          </div>

          <div
            className="flex-shrink-0 p-4 space-y-2"
            style={{ borderTop: `1px solid ${THEME.border}` }}
          >
            <HealthIndicator label="Battery" value={92} status="good" />
          </div>
        </div>
      </div>
    </div>
  );
}
