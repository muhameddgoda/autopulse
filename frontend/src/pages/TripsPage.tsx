import { useState, useEffect, useMemo } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from "recharts";
import {
  Car,
  Calendar,
  Clock,
  MapPin,
  Download,
  ChevronDown,
  ChevronUp,
  Route,
  Gauge,
  Fuel,
  Zap,
} from "lucide-react";

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
  purple: "#a855f7",
  cyan: "#06b6d4",
};

const MODE_COLORS: Record<string, string> = {
  parked: "#6b7280",
  reverse: THEME.purple,
  city: THEME.cyan,
  highway: THEME.blue,
  sport: THEME.orange,
};

// ============================================
// TYPES
// ============================================
interface Trip {
  id: string;
  vehicle_id: string;
  start_time: string;
  end_time: string | null;
  start_lat: number | null;
  start_lon: number | null;
  end_lat: number | null;
  end_lon: number | null;
  distance_km: number | null;
  fuel_used_liters: number | null;
  avg_speed_kmh: number | null;
  max_speed_kmh: number | null;
  mode_parked_seconds: number | null;
  mode_city_seconds: number | null;
  mode_highway_seconds: number | null;
  mode_sport_seconds: number | null;
  mode_reverse_seconds: number | null;
  duration_seconds: number | null;
  is_active: boolean;
}

interface Vehicle {
  id: string;
  name: string;
  model: string;
  year: number;
}

interface WeeklyStats {
  total_distance_km: number;
  total_trips: number;
  avg_speed: number;
  max_speed: number;
  total_fuel_used: number;
  mode_breakdown: {
    city: number;
    highway: number;
    sport: number;
  };
}

// ============================================
// API
// ============================================
const API_BASE = "http://localhost:8000/api/telemetry";

async function getVehicle(): Promise<Vehicle | null> {
  try {
    const res = await fetch(`${API_BASE}/vehicles`);
    if (!res.ok) return null;
    const vehicles = await res.json();
    return vehicles.length > 0 ? vehicles[0] : null;
  } catch {
    return null;
  }
}

async function getTrips(vehicleId: string): Promise<Trip[]> {
  try {
    const res = await fetch(`${API_BASE}/trips/${vehicleId}?limit=100`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function getWeeklyStats(vehicleId: string): Promise<WeeklyStats | null> {
  try {
    const res = await fetch(`${API_BASE}/stats/weekly/${vehicleId}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function exportTripsCSV(vehicleId: string, vehicleName: string) {
  try {
    const res = await fetch(`${API_BASE}/export/trips-csv/${vehicleId}`);
    if (!res.ok) throw new Error("Export failed");

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trips_${vehicleName}_${
      new Date().toISOString().split("T")[0]
    }.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (err) {
    console.error("Export error:", err);
    alert("Failed to export trips");
  }
}

// ============================================
// HELPER FUNCTIONS
// ============================================
function formatDuration(seconds: number | null): string {
  if (seconds == null || seconds === 0) return "0m";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function formatDurationFromTimes(start: string, end: string | null): string {
  if (!end) return "In Progress";
  const duration = new Date(end).getTime() - new Date(start).getTime();
  const hours = Math.floor(duration / (1000 * 60 * 60));
  const minutes = Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60));
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

// ============================================
// COMPONENTS
// ============================================
function StatCard({
  icon: Icon,
  label,
  value,
  subValue,
  color,
}: {
  icon: typeof Car;
  label: string;
  value: string | number;
  subValue?: string;
  color?: string;
}) {
  return (
    <div
      className="rounded-xl p-4 transition-all duration-200"
      style={{
        backgroundColor: THEME.card,
        border: `1px solid ${THEME.border}`,
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon size={16} style={{ color: color || THEME.orange }} />
        <span
          className="text-xs uppercase tracking-wider"
          style={{ color: THEME.textMuted }}
        >
          {label}
        </span>
      </div>
      <div
        className="text-2xl font-bold"
        style={{ color: color || THEME.text }}
      >
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

function TripCard({
  trip,
  isExpanded,
  onToggle,
}: {
  trip: Trip;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const tripDate = new Date(trip.start_time);

  // Build mode data from individual fields
  const tripModeData = useMemo(() => {
    const modes = [
      {
        name: "City",
        value: trip.mode_city_seconds || 0,
        color: MODE_COLORS.city,
      },
      {
        name: "Highway",
        value: trip.mode_highway_seconds || 0,
        color: MODE_COLORS.highway,
      },
      {
        name: "Sport",
        value: trip.mode_sport_seconds || 0,
        color: MODE_COLORS.sport,
      },
      {
        name: "Reverse",
        value: trip.mode_reverse_seconds || 0,
        color: MODE_COLORS.reverse,
      },
    ];
    return modes.filter((m) => m.value > 0);
  }, [trip]);

  // Calculate fuel efficiency
  const fuelEfficiency = useMemo(() => {
    if (trip.fuel_used_liters && trip.distance_km && trip.distance_km > 0) {
      return (trip.fuel_used_liters / trip.distance_km) * 100; // L/100km
    }
    return null;
  }, [trip]);

  return (
    <div
      className="rounded-xl overflow-hidden transition-all duration-200"
      style={{
        backgroundColor: THEME.card,
        border: `1px solid ${isExpanded ? THEME.borderLight : THEME.border}`,
      }}
    >
      {/* Header */}
      <div
        className="p-4 cursor-pointer transition-colors"
        style={{
          backgroundColor: isExpanded ? THEME.cardHover : "transparent",
        }}
        onClick={onToggle}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center"
              style={{ backgroundColor: THEME.orangeGlow }}
            >
              <Car size={24} style={{ color: THEME.orange }} />
            </div>
            <div>
              <div className="font-semibold" style={{ color: THEME.text }}>
                {tripDate.toLocaleDateString("en-US", {
                  weekday: "long",
                  month: "short",
                  day: "numeric",
                })}
              </div>
              <div
                className="text-sm flex items-center gap-2"
                style={{ color: THEME.textMuted }}
              >
                <Clock size={14} />
                {tripDate.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
                {" - "}
                {trip.end_time
                  ? new Date(trip.end_time).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })
                  : "Now"}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Avg Speed */}
            {trip.avg_speed_kmh != null && (
              <div className="text-right hidden sm:block">
                <div
                  className="text-lg font-bold"
                  style={{ color: THEME.green }}
                >
                  {trip.avg_speed_kmh.toFixed(0)}
                </div>
                <div className="text-xs" style={{ color: THEME.textMuted }}>
                  km/h avg
                </div>
              </div>
            )}

            {/* Duration */}
            <div className="text-right hidden md:block">
              <div className="font-semibold" style={{ color: THEME.text }}>
                {formatDuration(trip.duration_seconds)}
              </div>
              <div className="text-xs" style={{ color: THEME.textMuted }}>
                Duration
              </div>
            </div>

            {/* Distance */}
            {trip.distance_km != null && (
              <div className="text-right hidden md:block">
                <div className="font-semibold" style={{ color: THEME.orange }}>
                  {trip.distance_km.toFixed(1)} km
                </div>
                <div className="text-xs" style={{ color: THEME.textMuted }}>
                  Distance
                </div>
              </div>
            )}

            {/* Active badge */}
            {trip.is_active && (
              <span
                className="px-2 py-1 text-xs rounded-full"
                style={{
                  backgroundColor: THEME.greenGlow,
                  color: THEME.green,
                }}
              >
                Active
              </span>
            )}

            {/* Expand icon */}
            {isExpanded ? (
              <ChevronUp size={20} style={{ color: THEME.textMuted }} />
            ) : (
              <ChevronDown size={20} style={{ color: THEME.textMuted }} />
            )}
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div
          className="p-4"
          style={{
            borderTop: `1px solid ${THEME.border}`,
            backgroundColor: THEME.bg,
          }}
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Trip Details */}
            <div className="space-y-4">
              <h4
                className="font-semibold flex items-center gap-2"
                style={{ color: THEME.textMuted }}
              >
                <Route size={16} style={{ color: THEME.orange }} />
                Trip Details
              </h4>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span style={{ color: THEME.textMuted }}>Start</span>
                  <span style={{ color: THEME.text }}>
                    {new Date(trip.start_time).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: THEME.textMuted }}>End</span>
                  <span style={{ color: THEME.text }}>
                    {trip.end_time
                      ? new Date(trip.end_time).toLocaleString()
                      : "In Progress"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: THEME.textMuted }}>Duration</span>
                  <span style={{ color: THEME.text }}>
                    {formatDuration(trip.duration_seconds)}
                  </span>
                </div>
                {trip.distance_km != null && (
                  <div className="flex justify-between">
                    <span style={{ color: THEME.textMuted }}>Distance</span>
                    <span style={{ color: THEME.orange }}>
                      {trip.distance_km.toFixed(2)} km
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Speed & Fuel */}
            <div className="space-y-4">
              <h4
                className="font-semibold flex items-center gap-2"
                style={{ color: THEME.textMuted }}
              >
                <Zap size={16} style={{ color: THEME.orange }} />
                Speed & Fuel
              </h4>

              <div className="space-y-2 text-sm">
                {trip.avg_speed_kmh != null && (
                  <div className="flex justify-between">
                    <span style={{ color: THEME.textMuted }}>Avg Speed</span>
                    <span style={{ color: THEME.green }}>
                      {trip.avg_speed_kmh.toFixed(1)} km/h
                    </span>
                  </div>
                )}
                {trip.max_speed_kmh != null && (
                  <div className="flex justify-between">
                    <span style={{ color: THEME.textMuted }}>Max Speed</span>
                    <span style={{ color: THEME.blue }}>
                      {trip.max_speed_kmh.toFixed(1)} km/h
                    </span>
                  </div>
                )}
                {trip.fuel_used_liters != null && (
                  <div className="flex justify-between">
                    <span style={{ color: THEME.textMuted }}>Fuel Used</span>
                    <span style={{ color: THEME.yellow }}>
                      {trip.fuel_used_liters.toFixed(2)} L
                    </span>
                  </div>
                )}
                {fuelEfficiency != null && (
                  <div className="flex justify-between">
                    <span style={{ color: THEME.textMuted }}>Efficiency</span>
                    <span style={{ color: THEME.cyan }}>
                      {fuelEfficiency.toFixed(1)} L/100km
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Mode Breakdown & Location */}
            <div className="space-y-4">
              {/* Driving Modes */}
              <div>
                <h4
                  className="font-semibold flex items-center gap-2 mb-3"
                  style={{ color: THEME.textMuted }}
                >
                  <Gauge size={16} style={{ color: THEME.orange }} />
                  Driving Modes
                </h4>

                {tripModeData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={120}>
                    <PieChart>
                      <Pie
                        data={tripModeData}
                        cx="50%"
                        cy="50%"
                        innerRadius={30}
                        outerRadius={45}
                        dataKey="value"
                      >
                        {tripModeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Legend
                        formatter={(value) => (
                          <span style={{ color: THEME.text, fontSize: 11 }}>
                            {value}
                          </span>
                        )}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-sm" style={{ color: THEME.textMuted }}>
                    No mode data
                  </p>
                )}
              </div>

              {/* Location */}
              <div>
                <h4
                  className="font-semibold flex items-center gap-2 mb-2"
                  style={{ color: THEME.textMuted }}
                >
                  <MapPin size={16} style={{ color: THEME.orange }} />
                  Location
                </h4>

                <div className="space-y-1 text-sm">
                  {trip.start_lat && trip.start_lon ? (
                    <div>
                      <span style={{ color: THEME.textMuted }}>Start: </span>
                      <span style={{ color: THEME.text }}>
                        {trip.start_lat.toFixed(4)}, {trip.start_lon.toFixed(4)}
                      </span>
                    </div>
                  ) : null}
                  {trip.end_lat && trip.end_lon ? (
                    <div>
                      <span style={{ color: THEME.textMuted }}>End: </span>
                      <span style={{ color: THEME.text }}>
                        {trip.end_lat.toFixed(4)}, {trip.end_lon.toFixed(4)}
                      </span>
                    </div>
                  ) : null}
                  {!trip.start_lat && !trip.end_lat && (
                    <p style={{ color: THEME.textMuted }}>Not available</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// MAIN COMPONENT
// ============================================
export default function TripsPage() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [trips, setTrips] = useState<Trip[]>([]);
  const [weeklyStats, setWeeklyStats] = useState<WeeklyStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedTrip, setExpandedTrip] = useState<string | null>(null);

  // Load data
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      const v = await getVehicle();
      setVehicle(v);

      if (v) {
        const [tripsData, statsData] = await Promise.all([
          getTrips(v.id),
          getWeeklyStats(v.id),
        ]);
        setTrips(tripsData);
        setWeeklyStats(statsData);
      }

      setLoading(false);
    }
    loadData();
  }, []);

  // Mode breakdown for weekly stats
  const weeklyModeData = useMemo(() => {
    if (!weeklyStats?.mode_breakdown) return [];
    return Object.entries(weeklyStats.mode_breakdown)
      .filter(([_, value]) => value > 0)
      .map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        color: MODE_COLORS[name] || THEME.textMuted,
      }));
  }, [weeklyStats]);

  // Calculate overall stats from trips
  const overallStats = useMemo(() => {
    const completedTrips = trips.filter((t) => !t.is_active);
    const totalDistance = completedTrips.reduce(
      (sum, t) => sum + (t.distance_km || 0),
      0
    );
    const totalFuel = completedTrips.reduce(
      (sum, t) => sum + (t.fuel_used_liters || 0),
      0
    );
    const totalDuration = completedTrips.reduce(
      (sum, t) => sum + (t.duration_seconds || 0),
      0
    );

    // Calculate weighted average speed
    const tripsWithSpeed = completedTrips.filter(
      (t) => t.avg_speed_kmh != null && t.duration_seconds
    );
    const weightedSpeedSum = tripsWithSpeed.reduce(
      (sum, t) => sum + t.avg_speed_kmh! * (t.duration_seconds || 1),
      0
    );
    const totalWeightedTime = tripsWithSpeed.reduce(
      (sum, t) => sum + (t.duration_seconds || 1),
      0
    );
    const avgSpeed =
      totalWeightedTime > 0 ? weightedSpeedSum / totalWeightedTime : 0;

    return {
      totalTrips: completedTrips.length,
      totalDistance,
      totalFuel,
      totalDuration,
      avgSpeed,
    };
  }, [trips]);

  if (loading) {
    return (
      <div
        className="h-full flex items-center justify-center"
        style={{ backgroundColor: THEME.bg }}
      >
        <div className="text-center">
          <div
            className="w-12 h-12 border-4 border-t-orange-500 rounded-full animate-spin mx-auto mb-4"
            style={{ borderColor: THEME.border, borderTopColor: THEME.orange }}
          />
          <div style={{ color: THEME.textMuted }}>Loading trips...</div>
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
            Trip History
          </h1>
          <p className="text-xs mt-0.5" style={{ color: THEME.textMuted }}>
            {vehicle?.model || "Vehicle"} • {trips.length} trips recorded
          </p>
        </div>

        <button
          onClick={() => vehicle && exportTripsCSV(vehicle.id, vehicle.name)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors hover:opacity-90"
          style={{
            backgroundColor: THEME.orange,
            color: "#000",
          }}
        >
          <Download size={18} />
          Export CSV
        </button>
      </div>

      {/* Stats Section */}
      <div className="flex-shrink-0 p-6 pb-4">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <StatCard
            icon={Calendar}
            label="Total Trips"
            value={overallStats.totalTrips}
            subValue="completed"
          />
          <StatCard
            icon={Route}
            label="Distance"
            value={`${overallStats.totalDistance.toFixed(1)}`}
            subValue="km total"
          />
          <StatCard
            icon={Gauge}
            label="Avg Speed"
            value={`${overallStats.avgSpeed.toFixed(0)}`}
            subValue="km/h overall"
            color={THEME.green}
          />
          <StatCard
            icon={Fuel}
            label="Fuel Used"
            value={`${overallStats.totalFuel.toFixed(1)}`}
            subValue="liters total"
            color={THEME.yellow}
          />

          {/* Weekly Mode Breakdown */}
          <div
            className="rounded-xl p-4"
            style={{
              backgroundColor: THEME.card,
              border: `1px solid ${THEME.border}`,
            }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Zap size={16} style={{ color: THEME.orange }} />
              <span
                className="text-xs uppercase tracking-wider"
                style={{ color: THEME.textMuted }}
              >
                Modes
              </span>
            </div>
            {weeklyModeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={60}>
                <PieChart>
                  <Pie
                    data={weeklyModeData}
                    cx="50%"
                    cy="50%"
                    innerRadius={15}
                    outerRadius={25}
                    dataKey="value"
                  >
                    {weeklyModeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-xs" style={{ color: THEME.textMuted }}>
                No data
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Trips List Header */}
      <div
        className="flex-shrink-0 px-6 py-3 flex items-center justify-between"
        style={{ borderTop: `1px solid ${THEME.border}` }}
      >
        <h2
          className="text-sm font-semibold flex items-center gap-2"
          style={{ color: THEME.text }}
        >
          <Car size={16} style={{ color: THEME.orange }} />
          All Trips
        </h2>
        <span
          className="text-xs px-2 py-1 rounded"
          style={{ backgroundColor: THEME.card, color: THEME.textMuted }}
        >
          {trips.length} trips
        </span>
      </div>

      {/* Scrollable Trips List */}
      <div className="flex-1 overflow-y-auto px-6 pb-6">
        {trips.length === 0 ? (
          <div
            className="rounded-xl p-8 text-center"
            style={{
              backgroundColor: THEME.card,
              border: `1px solid ${THEME.border}`,
            }}
          >
            <Car
              size={48}
              className="mx-auto mb-4 opacity-50"
              style={{ color: THEME.textMuted }}
            />
            <p style={{ color: THEME.textMuted }}>No trips recorded yet</p>
            <p className="text-sm mt-2" style={{ color: THEME.textDim }}>
              Start the simulator to begin recording trips
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {trips.map((trip) => (
              <TripCard
                key={trip.id}
                trip={trip}
                isExpanded={expandedTrip === trip.id}
                onToggle={() =>
                  setExpandedTrip(expandedTrip === trip.id ? null : trip.id)
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
