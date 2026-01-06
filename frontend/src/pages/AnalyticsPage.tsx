import { useEffect, useMemo, useState } from 'react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { telemetryApi, vehicleApi } from '../lib/api';
import { Vehicle } from '../types';

// Theme colors
const ORANGE = '#f97316';
const DARK_BG = '#0a0a0a';
const DARK_CARD = '#141414';
const DARK_BORDER = '#262626';
const DARK_TEXT = '#ffffff';
const DARK_TEXT_MUTED = '#737373';

// Behavior colors
const BEHAVIOR_COLORS: Record<string, string> = {
  exemplary: '#22c55e',
  calm: '#3b82f6',
  normal: '#f59e0b',
  aggressive: '#f97316',
  dangerous: '#ef4444',
};

// Risk colors
const RISK_COLORS: Record<string, string> = {
  low: '#22c55e',
  medium: '#f59e0b',
  high: '#f97316',
  critical: '#ef4444',
};

type Scope = 'trip' | '7d' | '30d';

interface TripScore {
  score: number;
  behavior: string;
  risk_level: string;
  components?: Record<string, { score: number; weight: number; weighted: number }>;
  insights?: string[];
  recommendations?: string[];
  summary?: {
    duration_minutes: number;
    distance_km: number;
    avg_speed_kmh: number;
    max_speed_kmh: number;
    harsh_events_total: number;
    speeding_percentage: number;
  };
  ml_enhanced?: boolean;
  ml_behavior?: {
    prediction: string;
    confidence: Record<string, number>;
  };
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

// Compact Score Gauge - Fixed arc
function ScoreGauge({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return '#22c55e';
    if (s >= 60) return '#f59e0b';
    if (s >= 40) return '#f97316';
    return '#ef4444';
  };

  const color = getColor(score);
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degrees
  const filledLength = (score / 100) * arcLength;

  return (
    <div className="relative w-32 h-32">
      <svg viewBox="0 0 100 100" className="w-full h-full">
        {/* Background arc */}
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={DARK_BORDER}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${arcLength} ${circumference}`}
          transform="rotate(135 50 50)"
        />
        {/* Score arc */}
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${filledLength} ${circumference}`}
          transform="rotate(135 50 50)"
          style={{ transition: 'stroke-dasharray 0.5s ease' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold" style={{ color }}>
          {score.toFixed(0)}
        </span>
        <span className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>
          / 100
        </span>
      </div>
    </div>
  );
}

// Compact Component Card
function ComponentCard({ name, score, weight }: { name: string; score: number; weight: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return '#22c55e';
    if (s >= 60) return '#f59e0b';
    if (s >= 40) return '#f97316';
    return '#ef4444';
  };
  const color = getColor(score);
  const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());

  return (
    <div className="rounded-lg p-2.5" style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs truncate" style={{ color: DARK_TEXT }}>
          {displayName}
        </span>
        <span
          className="text-[10px] px-1.5 py-0.5 rounded"
          style={{ backgroundColor: `${color}20`, color }}
        >
          {(weight * 100).toFixed(0)}%
        </span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-lg font-bold" style={{ color }}>
          {score.toFixed(0)}
        </span>
        <span className="text-[10px]" style={{ color: DARK_TEXT_MUTED }}>
          / 100
        </span>
      </div>
      <div className="mt-1 h-1 rounded-full overflow-hidden" style={{ backgroundColor: DARK_BORDER }}>
        <div className="h-full rounded-full" style={{ width: `${score}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

// ML Confidence Chart - Horizontal bars
function MLConfidenceChart({ confidence }: { confidence: Record<string, number> }) {
  const data = Object.entries(confidence)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value: value * 100,
      fill: BEHAVIOR_COLORS[name] || ORANGE,
    }));

  return (
    <div className="h-24">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 10 }}>
          <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9, fill: DARK_TEXT_MUTED }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: DARK_TEXT_MUTED }} width={65} />
          <Tooltip
            contentStyle={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}`, fontSize: 11 }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Confidence']}
          />
          <Bar dataKey="value" radius={[0, 3, 3, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Radar Chart for Components
function ComponentsRadar({ components }: { components: Record<string, { score: number }> }) {
  const data = Object.entries(components).map(([name, { score }]) => ({
    subject: name.split('_').map((w) => w[0].toUpperCase()).join(''),
    fullName: name.replace(/_/g, ' '),
    score,
  }));

  return (
    <div className="h-44">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data}>
          <PolarGrid stroke={DARK_BORDER} />
          <PolarAngleAxis dataKey="subject" tick={{ fontSize: 9, fill: DARK_TEXT_MUTED }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 8, fill: DARK_TEXT_MUTED }} />
          <Radar dataKey="score" stroke={ORANGE} fill={ORANGE} fillOpacity={0.3} />
          <Tooltip
            contentStyle={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}`, fontSize: 11 }}
            formatter={(value: number, _name: any, props: any) => [`${Number(value).toFixed(0)}/100`, props?.payload?.fullName]}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

function deriveRisk(score: number): string {
  if (score >= 80) return 'low';
  if (score >= 60) return 'medium';
  if (score >= 40) return 'high';
  return 'critical';
}

export default function AnalyticsPage() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);

  const [scope, setScope] = useState<Scope>('trip');

  const [tripScore, setTripScore] = useState<TripScore | null>(null);
  const [rangeSummary, setRangeSummary] = useState<DriverSummary | null>(null);
  const [baseline30, setBaseline30] = useState<DriverSummary | null>(null);

  const [selectedTrip, setSelectedTrip] = useState<string | null>(null);
  const [trips, setTrips] = useState<any[]>([]);

  const scopeLabel = useMemo(() => {
    if (scope === 'trip') return 'Single Trip';
    if (scope === '7d') return 'Last 7 days';
    return 'Last 30 days';
  }, [scope]);

  useEffect(() => {
    async function fetchVehicle() {
      try {
        const vehicles = await vehicleApi.getAll();
        if (vehicles.length > 0) setVehicle(vehicles[0]);
      } catch (err) {
        console.error('Failed to fetch vehicle:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchVehicle();
  }, []);

  useEffect(() => {
    if (!vehicle) return;

    async function fetchTripsAndBaseline() {
      try {
        const tripsRes = await fetch(`http://localhost:8000/api/telemetry/trips/${vehicle.id}?limit=50`);
        const tripsData = await tripsRes.json();
        const completedTrips = tripsData.filter((t: any) => !t.is_active);
        setTrips(completedTrips);
        if (completedTrips.length > 0 && !selectedTrip) setSelectedTrip(completedTrips[0].id);

        // Preserve your old behavior: always show a 30-day baseline summary card (when available)
        try {
          const s30 = await telemetryApi.getDriverSummary(vehicle.id, 30);
          setBaseline30(s30);
        } catch (e) {
          // non-fatal
          console.warn('30-day baseline summary failed:', e);
        }
      } catch (err) {
        console.error('Failed to fetch trips:', err);
      }
    }

    fetchTripsAndBaseline();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vehicle]);

  useEffect(() => {
    if (!vehicle) return;

    async function loadScopeData() {
      try {
        if (scope === 'trip') {
          setRangeSummary(null);
          if (!selectedTrip) return;

          const hybridRes = await fetch(`http://localhost:8000/api/telemetry/ml/score/hybrid/${selectedTrip}`, { method: 'POST' });
          if (!hybridRes.ok) throw new Error('Hybrid score failed');
          const hybridData = await hybridRes.json();

          const analysisRes = await fetch(`http://localhost:8000/api/telemetry/ml/score/trip/${selectedTrip}`);
          if (analysisRes.ok) {
            const analysisData = await analysisRes.json();
            setTripScore({ ...analysisData, ...hybridData });
          } else {
            setTripScore(hybridData);
          }
        } else {
          const days = scope === '7d' ? 7 : 30;
          const summary = await telemetryApi.getDriverSummary(vehicle.id, days);
          setRangeSummary(summary);

          const avg = summary.score_statistics.average ?? 0;
          setTripScore({
            score: avg,
            behavior: summary.overall_behavior,
            risk_level: deriveRisk(avg),
            summary: {
              duration_minutes: (summary.totals.duration_hours ?? 0) * 60,
              distance_km: summary.totals.distance_km ?? 0,
              avg_speed_kmh: 0,
              max_speed_kmh: 0,
              harsh_events_total: (summary.totals.harsh_brakes ?? 0) + (summary.totals.harsh_accelerations ?? 0),
              speeding_percentage: 0,
            },
            insights: [],
            recommendations: [],
          });
        }
      } catch (err) {
        console.error('Failed to load analytics:', err);
      }
    }

    loadScopeData();
  }, [vehicle, selectedTrip, scope]);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center" style={{ backgroundColor: DARK_BG }}>
        <div style={{ color: DARK_TEXT_MUTED }}>Loading...</div>
      </div>
    );
  }

  return (
    <div className="h-full p-4 overflow-hidden flex flex-col" style={{ backgroundColor: DARK_BG }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold" style={{ color: DARK_TEXT }}>
            Driver Analytics
          </h1>
          <p className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
            {scope === 'trip' ? 'ML-powered behavior analysis & scoring' : `${scopeLabel} evaluation (average)`}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Scope selector */}
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value as Scope)}
            className="rounded-lg px-3 py-1.5 text-sm outline-none"
            style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}`, color: DARK_TEXT }}
          >
            <option value="trip">Single Trip</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
          </select>

          {/* Trip selector (trip mode only) */}
          {scope === 'trip' && (
            <select
              value={selectedTrip || ''}
              onChange={(e) => setSelectedTrip(e.target.value)}
              className="rounded-lg px-3 py-1.5 text-sm outline-none"
              style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}`, color: DARK_TEXT }}
            >
              <option value="">Select trip</option>
              {trips.map((trip) => (
                <option key={trip.id} value={trip.id}>
                  {new Date(trip.start_time).toLocaleDateString()} - {trip.distance_km?.toFixed(1) || '?'} km
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Main Content */}
      {tripScore ? (
        <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
          {/* Left Column */}
          <div className="col-span-3 flex flex-col gap-3">
            {/* Score Card */}
            <div
              className="rounded-xl p-4 flex flex-col items-center"
              style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
            >
              <ScoreGauge score={tripScore.score} />
              <div className="mt-2 flex items-center gap-2">
                <span
                  className="px-2 py-0.5 rounded-full text-xs font-medium"
                  style={{
                    backgroundColor: `${BEHAVIOR_COLORS[tripScore.behavior] || ORANGE}20`,
                    color: BEHAVIOR_COLORS[tripScore.behavior] || ORANGE,
                  }}
                >
                  {(tripScore.behavior || 'unknown').toUpperCase()}
                </span>
                <span
                  className="px-2 py-0.5 rounded-full text-xs"
                  style={{
                    backgroundColor: `${RISK_COLORS[tripScore.risk_level] || ORANGE}20`,
                    color: RISK_COLORS[tripScore.risk_level] || ORANGE,
                  }}
                >
                  {(tripScore.risk_level || 'unknown').toUpperCase()}
                </span>
              </div>
              {tripScore.ml_enhanced && scope === 'trip' && (
                <div className="mt-1 flex items-center gap-1 text-[10px]" style={{ color: '#22c55e' }}>
                  <span>✨</span>
                  <span>ML Enhanced</span>
                </div>
              )}
            </div>

            {/* Trip Summary (trip mode) OR Range Summary (aggregate modes) */}
            {scope === 'trip' ? (
              <div className="rounded-xl p-3" style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}>
                <h3 className="text-xs font-semibold mb-2" style={{ color: DARK_TEXT }}>
                  Trip Summary
                </h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div style={{ color: DARK_TEXT_MUTED }}>Duration</div>
                    <div className="font-semibold" style={{ color: DARK_TEXT }}>
                      {tripScore.summary?.duration_minutes?.toFixed(0) || '?'} min
                    </div>
                  </div>
                  <div>
                    <div style={{ color: DARK_TEXT_MUTED }}>Distance</div>
                    <div className="font-semibold" style={{ color: DARK_TEXT }}>
                      {tripScore.summary?.distance_km?.toFixed(1) || '?'} km
                    </div>
                  </div>
                  <div>
                    <div style={{ color: DARK_TEXT_MUTED }}>Avg Speed</div>
                    <div className="font-semibold" style={{ color: DARK_TEXT }}>
                      {tripScore.summary?.avg_speed_kmh?.toFixed(0) || '?'} km/h
                    </div>
                  </div>
                  <div>
                    <div style={{ color: DARK_TEXT_MUTED }}>Max Speed</div>
                    <div className="font-semibold" style={{ color: DARK_TEXT }}>
                      {tripScore.summary?.max_speed_kmh?.toFixed(0) || '?'} km/h
                    </div>
                  </div>
                  <div>
                    <div style={{ color: DARK_TEXT_MUTED }}>Harsh Events</div>
                    <div className="font-semibold" style={{ color: ORANGE }}>
                      {tripScore.summary?.harsh_events_total || 0}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: DARK_TEXT_MUTED }}>Speeding</div>
                    <div
                      className="font-semibold"
                      style={{ color: (tripScore.summary?.speeding_percentage || 0) > 20 ? '#ef4444' : DARK_TEXT }}
                    >
                      {tripScore.summary?.speeding_percentage?.toFixed(0) || '0'}%
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-xl p-3" style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}>
                <h3 className="text-xs font-semibold mb-2" style={{ color: DARK_TEXT }}>
                  {scopeLabel} Summary
                </h3>
                {rangeSummary ? (
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between">
                      <span style={{ color: DARK_TEXT_MUTED }}>Total Trips</span>
                      <span className="font-semibold" style={{ color: DARK_TEXT }}>
                        {rangeSummary.total_trips}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: DARK_TEXT_MUTED }}>Avg Score</span>
                      <span className="font-semibold" style={{ color: ORANGE }}>
                        {rangeSummary.score_statistics.average.toFixed(0)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: DARK_TEXT_MUTED }}>Total Distance</span>
                      <span className="font-semibold" style={{ color: DARK_TEXT }}>
                        {rangeSummary.totals.distance_km.toFixed(0)} km
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: DARK_TEXT_MUTED }}>Driving Time</span>
                      <span className="font-semibold" style={{ color: DARK_TEXT }}>
                        {rangeSummary.totals.duration_hours.toFixed(1)} hrs
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: DARK_TEXT_MUTED }}>Trend</span>
                      <span
                        className="font-semibold"
                        style={{
                          color:
                            rangeSummary.trend === 'improving'
                              ? '#22c55e'
                              : rangeSummary.trend === 'declining'
                                ? '#ef4444'
                                : DARK_TEXT_MUTED,
                        }}
                      >
                        {rangeSummary.trend === 'improving'
                          ? '📈 Improving'
                          : rangeSummary.trend === 'declining'
                            ? '📉 Declining'
                            : '➡️ Stable'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span style={{ color: DARK_TEXT_MUTED }}>Harsh Events/100km</span>
                      <span className="font-semibold" style={{ color: DARK_TEXT }}>
                        {(rangeSummary.events_per_100km.harsh_brakes + rangeSummary.events_per_100km.harsh_accels).toFixed(1)}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div style={{ color: DARK_TEXT_MUTED }} className="text-xs">
                    No summary available.
                  </div>
                )}
              </div>
            )}

            {/* ML Behavior Confidence (trip mode only) */}
            {scope === 'trip' && tripScore.ml_behavior && (
              <div className="rounded-xl p-3" style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}>
                <h3 className="text-xs font-semibold mb-2" style={{ color: DARK_TEXT }}>
                  ML Confidence
                </h3>
                <MLConfidenceChart confidence={tripScore.ml_behavior.confidence} />
              </div>
            )}
          </div>

          {/* Middle Column */}
          <div className="col-span-5 flex flex-col gap-3 min-h-0">
            {scope === 'trip' && tripScore.components ? (
              <>
                {/* Components Radar */}
                <div className="rounded-xl p-3" style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}>
                  <h3 className="text-xs font-semibold mb-1" style={{ color: DARK_TEXT }}>
                    Scoring Components
                  </h3>
                  <ComponentsRadar components={tripScore.components} />
                </div>

                {/* Component Cards Grid */}
                <div className="grid grid-cols-3 gap-2 flex-1 min-h-0 overflow-auto pr-1">
                  {Object.entries(tripScore.components).map(([name, data]) => (
                    <ComponentCard key={name} name={name} score={data.score} weight={data.weight} />
                  ))}
                </div>
              </>
            ) : (
              <div
                className="rounded-xl p-6 flex flex-col items-center justify-center"
                style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
              >
                <div className="text-4xl mb-2">📈</div>
                <div className="text-sm font-semibold" style={{ color: DARK_TEXT }}>
                  Aggregate view
                </div>
                <div className="text-xs text-center mt-1" style={{ color: DARK_TEXT_MUTED }}>
                  Component breakdown is shown per-trip. Switch to “Single Trip” for full explainability.
                </div>
              </div>
            )}
          </div>

          {/* Right Column */}
          <div className="col-span-4 flex flex-col gap-3 min-h-0">
            {/* Insights & Recommendations (trip mode only) */}
            {scope === 'trip' && (
              <div className="rounded-xl p-3 flex-1 min-h-0 overflow-auto" style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}>
                <h3 className="text-xs font-semibold mb-2" style={{ color: DARK_TEXT }}>
                  Insights & Recommendations
                </h3>

                {tripScore.insights?.length ? (
                  <div className="mb-3">
                    <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: DARK_TEXT_MUTED }}>
                      Insights
                    </div>
                    <ul className="space-y-1">
                      {tripScore.insights.slice(0, 4).map((insight, i) => (
                        <li key={i} className="flex items-start gap-1.5 text-xs" style={{ color: DARK_TEXT }}>
                          <span className="mt-1 w-1 h-1 rounded-full flex-shrink-0" style={{ backgroundColor: ORANGE }} />
                          {insight}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {tripScore.recommendations?.length ? (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: DARK_TEXT_MUTED }}>
                      Recommendations
                    </div>
                    <ul className="space-y-1">
                      {tripScore.recommendations.slice(0, 2).map((rec, i) => (
                        <li key={i} className="flex items-start gap-1.5 text-xs" style={{ color: '#3b82f6' }}>
                          <span className="mt-0.5">💡</span>
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            )}

            {/* Baseline 30-day summary (trip mode) */}
            {scope === 'trip' && baseline30 && baseline30.total_trips > 0 && (
              <div className="rounded-xl p-3" style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}>
                <h3 className="text-xs font-semibold mb-2" style={{ color: DARK_TEXT }}>
                  30-Day Summary
                </h3>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <span style={{ color: DARK_TEXT_MUTED }}>Total Trips</span>
                    <span className="font-semibold" style={{ color: DARK_TEXT }}>
                      {baseline30.total_trips}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: DARK_TEXT_MUTED }}>Avg Score</span>
                    <span className="font-semibold" style={{ color: ORANGE }}>
                      {baseline30.score_statistics.average.toFixed(0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: DARK_TEXT_MUTED }}>Total Distance</span>
                    <span className="font-semibold" style={{ color: DARK_TEXT }}>
                      {baseline30.totals.distance_km.toFixed(0)} km
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: DARK_TEXT_MUTED }}>Driving Time</span>
                    <span className="font-semibold" style={{ color: DARK_TEXT }}>
                      {baseline30.totals.duration_hours.toFixed(1)} hrs
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: DARK_TEXT_MUTED }}>Trend</span>
                    <span
                      className="font-semibold"
                      style={{
                        color:
                          baseline30.trend === 'improving'
                            ? '#22c55e'
                            : baseline30.trend === 'declining'
                              ? '#ef4444'
                              : DARK_TEXT_MUTED,
                      }}
                    >
                      {baseline30.trend === 'improving'
                        ? '📈 Improving'
                        : baseline30.trend === 'declining'
                          ? '📉 Declining'
                          : '➡️ Stable'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: DARK_TEXT_MUTED }}>Harsh Events/100km</span>
                    <span className="font-semibold" style={{ color: DARK_TEXT }}>
                      {(baseline30.events_per_100km.harsh_brakes + baseline30.events_per_100km.harsh_accels).toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Range summary (aggregate modes) - already shown on left; keep right side clean */}
          </div>
        </div>
      ) : (
        <div
          className="flex-1 rounded-xl p-8 flex flex-col items-center justify-center"
          style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
        >
          <div className="text-5xl mb-3">📊</div>
          <h2 className="text-lg font-semibold mb-1" style={{ color: DARK_TEXT }}>
            No Data
          </h2>
          <p className="text-xs text-center max-w-sm" style={{ color: DARK_TEXT_MUTED }}>
            Select a completed trip (Single Trip mode), or choose an evaluation window (7d/30d).
          </p>
        </div>
      )}
    </div>
  );
}
