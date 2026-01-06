import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { Route, Play, Square, Clock, Download, FileText, Brain, TrendingUp, Fuel, Gauge, AlertTriangle } from 'lucide-react';
import { useTelemetry } from '../hooks/useTelemetry';
import { vehicleApi, tripApi, WeeklyStats } from '../lib/api';
import { Vehicle, Trip } from '../types';

// Theme colors
const ORANGE = '#f97316';
const DARK_BG = '#0a0a0a';
const DARK_CARD = '#141414';
const DARK_BORDER = '#262626';
const DARK_TEXT = '#ffffff';
const DARK_TEXT_MUTED = '#737373';

const MODE_COLORS = {
  parked: '#6b7280',
  city: '#06b6d4',
  highway: '#3b82f6',
  sport: '#f97316',
  reverse: '#a855f7',
};

// Format duration
function formatDuration(seconds: number): string {
  if (!seconds) return '0m';
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  if (hrs > 0) return `${hrs}h ${mins}m`;
  return `${mins}m`;
}

// Mode Breakdown Chart
function ModeBreakdownChart({ data }: { data: Record<string, number> }) {
  const chartData = Object.entries(data)
    .filter(([_, seconds]) => seconds > 0)
    .map(([mode, seconds]) => ({
      name: mode.charAt(0).toUpperCase() + mode.slice(1),
      value: seconds,
      color: MODE_COLORS[mode as keyof typeof MODE_COLORS] || '#666',
    }));

  if (chartData.length === 0) {
    return <div className="text-sm text-center py-8" style={{ color: DARK_TEXT_MUTED }}>No data yet</div>;
  }

  return (
    <div className="flex items-center">
      <ResponsiveContainer width="50%" height={120}>
        <PieChart>
          <Pie data={chartData} cx="50%" cy="50%" innerRadius={30} outerRadius={50} paddingAngle={2} dataKey="value">
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.color} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="flex-1 space-y-1">
        {chartData.map((entry) => (
          <div key={entry.name} className="flex items-center gap-2 text-xs">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
            <span style={{ color: DARK_TEXT_MUTED }}>{entry.name}</span>
            <span className="ml-auto" style={{ color: DARK_TEXT_MUTED }}>{formatDuration(entry.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Trip Card
function TripCard({ trip, index }: { trip: Trip; index: number }) {
  const startDate = new Date(trip.start_time);
  const modeData = {
    parked: trip.mode_parked_seconds || 0,
    city: trip.mode_city_seconds || 0,
    highway: trip.mode_highway_seconds || 0,
    sport: trip.mode_sport_seconds || 0,
    reverse: trip.mode_reverse_seconds || 0,
  };
  const totalModeTime = Object.values(modeData).reduce((a, b) => a + b, 0);

  return (
    <div 
      className="rounded-xl p-4"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ backgroundColor: ORANGE + '20', color: ORANGE }}>
            #{index + 1}
          </span>
          <span className="text-sm font-medium" style={{ color: DARK_TEXT }}>
            {startDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
          </span>
          <span className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
            {startDate.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
        <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>{formatDuration(trip.duration_seconds || 0)}</div>
      </div>
      
      <div className="grid grid-cols-4 gap-3 mb-3 text-center">
        <div>
          <div className="text-lg font-semibold" style={{ color: DARK_TEXT }}>{(trip.distance_km || 0).toFixed(1)}</div>
          <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>km</div>
        </div>
        <div>
          <div className="text-lg font-semibold" style={{ color: DARK_TEXT }}>{Math.round(trip.avg_speed_kmh || 0)}</div>
          <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>avg km/h</div>
        </div>
        <div>
          <div className="text-lg font-semibold" style={{ color: DARK_TEXT }}>{Math.round(trip.max_speed_kmh || 0)}</div>
          <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>max km/h</div>
        </div>
        <div>
          <div className="text-lg font-semibold" style={{ color: DARK_TEXT }}>{(trip.fuel_used_liters || 0).toFixed(1)}</div>
          <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>L fuel</div>
        </div>
      </div>

      {/* Mode bar */}
      {totalModeTime > 0 && (
        <div className="flex h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: DARK_BORDER }}>
          {Object.entries(modeData).map(([mode, seconds]) => {
            if (seconds === 0) return null;
            return (
              <div 
                key={mode}
                style={{ width: `${(seconds / totalModeTime) * 100}%`, backgroundColor: MODE_COLORS[mode as keyof typeof MODE_COLORS] }}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

// ML Ideas Card
function MLInsightsCard({ trips }: { trips: Trip[] }) {
  // Calculate some basic insights from trip data
  const avgFuelEfficiency = trips.length > 0 
    ? trips.reduce((sum, t) => {
        const efficiency = (t.distance_km || 0) / ((t.fuel_used_liters || 0.01));
        return sum + efficiency;
      }, 0) / trips.length
    : 0;
  
  const sportModePercent = trips.length > 0
    ? (trips.reduce((sum, t) => sum + (t.mode_sport_seconds || 0), 0) / 
       trips.reduce((sum, t) => sum + (t.duration_seconds || 1), 0)) * 100
    : 0;

  const avgMaxRpm = trips.length > 0
    ? trips.reduce((sum, t) => sum + (t.max_rpm || 0), 0) / trips.length
    : 0;

  return (
    <div 
      className="rounded-2xl p-5"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-4 h-4" style={{ color: ORANGE }} />
        <span className="text-xs uppercase tracking-wider" style={{ color: DARK_TEXT_MUTED }}>ML-Ready Insights</span>
      </div>
      
      <div className="space-y-4">
        {/* Fuel Efficiency Prediction */}
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg" style={{ backgroundColor: ORANGE + '20' }}>
            <Fuel className="w-4 h-4" style={{ color: ORANGE }} />
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium" style={{ color: DARK_TEXT }}>Fuel Efficiency</div>
            <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
              Avg: {avgFuelEfficiency.toFixed(1)} km/L
            </div>
            <div className="text-[10px] mt-1 px-2 py-0.5 rounded inline-block" style={{ backgroundColor: '#22c55e20', color: '#22c55e' }}>
              Can predict optimal driving style
            </div>
          </div>
        </div>

        {/* Driving Behavior */}
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg" style={{ backgroundColor: '#3b82f620' }}>
            <TrendingUp className="w-4 h-4" style={{ color: '#3b82f6' }} />
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium" style={{ color: DARK_TEXT }}>Driving Behavior</div>
            <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
              Sport mode: {sportModePercent.toFixed(1)}% of time
            </div>
            <div className="text-[10px] mt-1 px-2 py-0.5 rounded inline-block" style={{ backgroundColor: '#3b82f620', color: '#3b82f6' }}>
              Can classify driver aggression
            </div>
          </div>
        </div>

        {/* Maintenance Prediction */}
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg" style={{ backgroundColor: '#f59e0b20' }}>
            <Gauge className="w-4 h-4" style={{ color: '#f59e0b' }} />
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium" style={{ color: DARK_TEXT }}>Engine Stress</div>
            <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
              Avg peak RPM: {avgMaxRpm.toFixed(0)}
            </div>
            <div className="text-[10px] mt-1 px-2 py-0.5 rounded inline-block" style={{ backgroundColor: '#f59e0b20', color: '#f59e0b' }}>
              Can predict maintenance needs
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t" style={{ borderColor: DARK_BORDER }}>
        <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>
          Export data to train models for: fuel optimization, predictive maintenance, driver scoring, route optimization
        </div>
      </div>
    </div>
  );
}

// Export Options Modal
function ExportModal({ vehicle, onClose }: { vehicle: Vehicle; onClose: () => void }) {
  const [exportType, setExportType] = useState<'trips' | 'telemetry'>('trips');
  const [days, setDays] = useState(30);
  const [hours, setHours] = useState(24);

  const handleExport = () => {
    if (exportType === 'trips') {
      window.open(`http://localhost:8000/api/telemetry/export/trips-csv/${vehicle.id}?days=${days}`, '_blank');
    } else {
      window.open(`http://localhost:8000/api/telemetry/export/csv/${vehicle.id}?hours=${hours}`, '_blank');
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div 
        className="w-full max-w-md rounded-2xl p-6"
        style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
      >
        <h3 className="text-lg font-bold mb-4" style={{ color: DARK_TEXT }}>Export Data</h3>
        
        {/* Export Type Selection */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setExportType('trips')}
            className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
              exportType === 'trips' ? 'text-white' : ''
            }`}
            style={{ 
              backgroundColor: exportType === 'trips' ? ORANGE : DARK_BG,
              color: exportType === 'trips' ? 'white' : DARK_TEXT_MUTED
            }}
          >
            Trip Summaries
          </button>
          <button
            onClick={() => setExportType('telemetry')}
            className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors`}
            style={{ 
              backgroundColor: exportType === 'telemetry' ? ORANGE : DARK_BG,
              color: exportType === 'telemetry' ? 'white' : DARK_TEXT_MUTED
            }}
          >
            Raw Telemetry
          </button>
        </div>

        {/* Time Range */}
        <div className="mb-4">
          <label className="text-xs mb-2 block" style={{ color: DARK_TEXT_MUTED }}>
            {exportType === 'trips' ? 'Days of data' : 'Hours of data'}
          </label>
          {exportType === 'trips' ? (
            <select 
              value={days} 
              onChange={(e) => setDays(Number(e.target.value))}
              className="w-full p-2 rounded-lg text-sm"
              style={{ backgroundColor: DARK_BG, color: DARK_TEXT, border: `1px solid ${DARK_BORDER}` }}
            >
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
              <option value={365}>Last year</option>
            </select>
          ) : (
            <select 
              value={hours} 
              onChange={(e) => setHours(Number(e.target.value))}
              className="w-full p-2 rounded-lg text-sm"
              style={{ backgroundColor: DARK_BG, color: DARK_TEXT, border: `1px solid ${DARK_BORDER}` }}
            >
              <option value={1}>Last 1 hour</option>
              <option value={6}>Last 6 hours</option>
              <option value={24}>Last 24 hours</option>
              <option value={48}>Last 48 hours</option>
              <option value={168}>Last week</option>
            </select>
          )}
        </div>

        {/* Data Description */}
        <div className="p-3 rounded-lg mb-4" style={{ backgroundColor: DARK_BG }}>
          <div className="text-xs font-medium mb-2" style={{ color: DARK_TEXT }}>
            {exportType === 'trips' ? 'Trip CSV includes:' : 'Telemetry CSV includes:'}
          </div>
          <div className="text-[10px] space-y-1" style={{ color: DARK_TEXT_MUTED }}>
            {exportType === 'trips' ? (
              <>
                <div>• Trip ID, start/end times, duration</div>
                <div>• Distance, avg/max speed, avg/max RPM</div>
                <div>• Fuel consumption</div>
                <div>• Driving mode breakdown (city/highway/sport)</div>
              </>
            ) : (
              <>
                <div>• Timestamp (1Hz resolution)</div>
                <div>• Speed, RPM, gear, throttle position</div>
                <div>• Engine temp, oil temp, oil pressure</div>
                <div>• Fuel level, battery voltage</div>
                <div>• GPS coordinates, driving mode</div>
              </>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-lg text-sm"
            style={{ backgroundColor: DARK_BG, color: DARK_TEXT_MUTED }}
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            className="flex-1 py-2 rounded-lg text-sm font-medium text-white flex items-center justify-center gap-2"
            style={{ backgroundColor: ORANGE }}
          >
            <Download className="w-4 h-4" />
            Download CSV
          </button>
        </div>
      </div>
    </div>
  );
}

export default function TripsPage() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTrip, setActiveTrip] = useState<Trip | null>(null);
  const [tripHistory, setTripHistory] = useState<Trip[]>([]);
  const [weeklyStats, setWeeklyStats] = useState<WeeklyStats | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);

  useEffect(() => {
    async function fetchVehicle() {
      try {
        const vehicles = await vehicleApi.getAll();
        if (vehicles.length > 0) setVehicle(vehicles[0]);
      } catch (error) {
        console.error('Failed to fetch vehicle:', error);
      } finally {
        setLoading(false);
      }
    }
    fetchVehicle();
  }, []);

  useEffect(() => {
    if (!vehicle) return;
    async function fetchTrips() {
      try {
        const [active, history] = await Promise.all([
          tripApi.getActive(vehicle!.id).catch(() => null),
          tripApi.getHistory(vehicle!.id, 20).catch(() => []),
        ]);
        setActiveTrip(active);
        setTripHistory(history.filter(t => !t.is_active));
        tripApi.getWeeklyStats(vehicle!.id).then(setWeeklyStats).catch(() => {});
      } catch (error) {
        console.error('Failed to fetch trips:', error);
      }
    }
    fetchTrips();
    const interval = setInterval(fetchTrips, 5000);
    return () => clearInterval(interval);
  }, [vehicle]);

  const { telemetry } = useTelemetry({
    vehicleId: vehicle?.id ?? null,
    enabled: !!vehicle,
  });

  const handleStartTrip = async () => {
    if (!vehicle) return;
    setActionLoading(true);
    try {
      const trip = await tripApi.start(vehicle.id);
      setActiveTrip(trip);
    } catch (error) {
      console.error('Failed to start trip:', error);
    } finally {
      setActionLoading(false);
    }
  };

  const handleEndTrip = async () => {
    if (!vehicle) return;
    setActionLoading(true);
    try {
      const endedTrip = await tripApi.endActiveTrip(vehicle.id);
      if (endedTrip) {
        setActiveTrip(null);
        setTripHistory(prev => [endedTrip, ...prev]);
      }
    } catch (error) {
      console.error('Failed to end trip:', error);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center" style={{ backgroundColor: DARK_BG }}>
        <div style={{ color: DARK_TEXT_MUTED }}>Loading...</div>
      </div>
    );
  }

  return (
    <div className="h-full p-6 overflow-auto" style={{ backgroundColor: DARK_BG }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: DARK_TEXT }}>Trip Analytics</h1>
          <p className="text-sm" style={{ color: DARK_TEXT_MUTED }}>Track, analyze, and export your driving data</p>
        </div>
        <button 
          onClick={() => setShowExportModal(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-colors hover:opacity-90"
          style={{ backgroundColor: ORANGE, color: 'white' }}
        >
          <Download className="w-4 h-4" />
          Export Data
        </button>
      </div>

      <div className="grid grid-cols-12 gap-5">
        {/* Active Trip / Start Button */}
        <div className="col-span-8">
          {activeTrip ? (
            <div 
              className="rounded-2xl p-5"
              style={{ backgroundColor: DARK_CARD, border: `2px solid ${ORANGE}`, boxShadow: `0 0 20px ${ORANGE}30` }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="font-semibold" style={{ color: DARK_TEXT }}>Trip in Progress</span>
                  <span className="text-xs px-2 py-0.5 rounded-full" style={{ backgroundColor: ORANGE + '20', color: ORANGE }}>
                    Recording
                  </span>
                </div>
                <button
                  onClick={handleEndTrip}
                  disabled={actionLoading}
                  className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg text-sm font-medium hover:bg-red-600 transition-colors disabled:opacity-50"
                >
                  <Square className="w-4 h-4" />
                  End Trip
                </button>
              </div>
              <div className="grid grid-cols-5 gap-4">
                <div className="text-center">
                  <div className="text-3xl font-bold" style={{ color: ORANGE }}>{Math.round(telemetry?.speed_kmh || 0)}</div>
                  <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>km/h</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold" style={{ color: DARK_TEXT }}>{Math.round(activeTrip.max_speed_kmh || 0)}</div>
                  <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>max km/h</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold" style={{ color: DARK_TEXT }}>{(activeTrip.max_rpm || 0).toLocaleString()}</div>
                  <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>max RPM</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold" style={{ color: DARK_TEXT }}>
                    {telemetry?.gear === -1 ? 'R' : telemetry?.gear === 0 ? 'N' : telemetry?.gear || 'P'}
                  </div>
                  <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>gear</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold" style={{ color: telemetry?.fuel_level && telemetry.fuel_level < 15 ? '#f59e0b' : DARK_TEXT }}>
                    {(telemetry?.fuel_level || 0).toFixed(0)}%
                  </div>
                  <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>fuel</div>
                </div>
              </div>
            </div>
          ) : (
            <button
              onClick={handleStartTrip}
              disabled={actionLoading}
              className="w-full py-8 rounded-2xl border-2 border-dashed transition-all flex flex-col items-center justify-center gap-2 disabled:opacity-50 hover:border-orange-500 group"
              style={{ borderColor: DARK_BORDER, color: DARK_TEXT_MUTED }}
            >
              <div className="p-4 rounded-full transition-colors group-hover:bg-orange-500/20" style={{ backgroundColor: DARK_BG }}>
                <Play className="w-8 h-8 group-hover:text-orange-500" />
              </div>
              <span className="text-lg font-medium group-hover:text-orange-500">Start New Trip</span>
              <span className="text-xs">Begin recording your drive</span>
            </button>
          )}
        </div>

        {/* Weekly Stats */}
        <div 
          className="col-span-4 rounded-2xl p-5"
          style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
        >
          <div className="text-xs uppercase tracking-wider mb-4" style={{ color: DARK_TEXT_MUTED }}>This Week</div>
          {weeklyStats ? (
            <>
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="text-center">
                  <div className="text-xl font-bold" style={{ color: DARK_TEXT }}>{weeklyStats.total_trips}</div>
                  <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>trips</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold" style={{ color: DARK_TEXT }}>{weeklyStats.total_distance_km.toFixed(1)}</div>
                  <div className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>km</div>
                </div>
              </div>
              <ModeBreakdownChart data={weeklyStats.mode_breakdown_seconds} />
            </>
          ) : (
            <div className="text-sm text-center py-8" style={{ color: DARK_TEXT_MUTED }}>No data yet</div>
          )}
        </div>

        {/* ML Insights */}
        <div className="col-span-4">
          <MLInsightsCard trips={tripHistory} />
        </div>

        {/* Trip History */}
        <div className="col-span-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" style={{ color: DARK_TEXT_MUTED }} />
              <span className="text-sm font-medium" style={{ color: DARK_TEXT }}>Recent Trips</span>
              <span className="text-xs px-2 py-0.5 rounded-full" style={{ backgroundColor: DARK_CARD, color: DARK_TEXT_MUTED }}>
                {tripHistory.length} recorded
              </span>
            </div>
          </div>
          
          {tripHistory.length === 0 ? (
            <div 
              className="rounded-2xl p-8 text-center"
              style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
            >
              <Route className="w-12 h-12 mx-auto mb-3" style={{ color: DARK_TEXT_MUTED }} />
              <p style={{ color: DARK_TEXT_MUTED }}>No trips recorded yet</p>
              <p className="text-sm" style={{ color: DARK_TEXT_MUTED }}>Start and complete a trip to see history</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4 max-h-[400px] overflow-y-auto pr-2">
              {tripHistory.map((trip, index) => (
                <TripCard key={trip.id} trip={trip} index={index} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Export Modal */}
      {showExportModal && vehicle && (
        <ExportModal vehicle={vehicle} onClose={() => setShowExportModal(false)} />
      )}
    </div>
  );
}
