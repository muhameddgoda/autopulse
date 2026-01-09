import { useState, useEffect, useCallback } from "react";
import {
  DARK_CARD,
  DARK_BORDER,
  DARK_TEXT,
  DARK_TEXT_MUTED,
  ORANGE,
} from "../../constants/theme";

interface TripInfoWidgetProps {
  vehicleId: string | null;
  onTripEnd?: () => void; // Callback when trip ends
}

export function TripInfoWidget({ vehicleId, onTripEnd }: TripInfoWidgetProps) {
  const [activeTrip, setActiveTrip] = useState<any>(null);
  const [elapsed, setElapsed] = useState(0);
  const [wasActive, setWasActive] = useState(false);

  const fetchTrip = useCallback(async () => {
    if (!vehicleId) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/telemetry/trips/active/${vehicleId}`
      );
      if (res.ok) {
        const trip = await res.json();
        // Check if trip exists AND is active
        if (trip && trip.is_active) {
          setActiveTrip(trip);
          setWasActive(true);
        } else {
          // Trip ended or no active trip
          if (wasActive && onTripEnd) {
            onTripEnd();
          }
          setActiveTrip(null);
          setElapsed(0);
          setWasActive(false);
        }
      } else {
        // No active trip found
        if (wasActive && onTripEnd) {
          onTripEnd();
        }
        setActiveTrip(null);
        setElapsed(0);
        setWasActive(false);
      }
    } catch (e) {
      console.error("Failed to fetch trip:", e);
    }
  }, [vehicleId, wasActive, onTripEnd]);

  useEffect(() => {
    fetchTrip();
    const interval = setInterval(fetchTrip, 3000); // Check more frequently
    return () => clearInterval(interval);
  }, [fetchTrip]);

  useEffect(() => {
    if (!activeTrip) return;
    const startTime = new Date(activeTrip.start_time).getTime();

    const updateElapsed = () => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    };

    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);
    return () => clearInterval(interval);
  }, [activeTrip]);

  const formatTime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) return `${hrs}h ${mins}m`;
    return `${mins}m ${secs}s`;
  };

  const formatStartTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  };

  if (!activeTrip) {
    return (
      <div
        className="rounded-2xl p-5 h-full"
        style={{
          backgroundColor: DARK_CARD,
          border: `1px solid ${DARK_BORDER}`,
        }}
      >
        <div
          className="text-xs uppercase tracking-wider mb-3"
          style={{ color: DARK_TEXT_MUTED }}
        >
          Current Trip
        </div>
        <div className="flex flex-col items-center justify-center py-6">
          <div className="text-4xl mb-2">🚗</div>
          <div className="text-sm" style={{ color: DARK_TEXT_MUTED }}>
            No active trip
          </div>
          <div className="text-xs mt-1" style={{ color: DARK_TEXT_MUTED }}>
            Start driving to begin
          </div>
        </div>
      </div>
    );
  }

  const avgSpeed =
    activeTrip.speed_sum && activeTrip.total_readings
      ? activeTrip.speed_sum / activeTrip.total_readings
      : 0;
  const distanceKm = (avgSpeed * elapsed) / 3600;

  return (
    <div
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div
        className="text-xs uppercase tracking-wider mb-3"
        style={{ color: DARK_TEXT_MUTED }}
      >
        Current Trip
      </div>
      <div className="flex items-center gap-2 mb-4">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        <span className="text-sm" style={{ color: DARK_TEXT_MUTED }}>
          Started at {formatStartTime(activeTrip.start_time)}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-2xl font-bold" style={{ color: DARK_TEXT }}>
            {formatTime(elapsed)}
          </div>
          <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
            Duration
          </div>
        </div>
        <div>
          <div className="text-2xl font-bold" style={{ color: ORANGE }}>
            {distanceKm.toFixed(1)} km
          </div>
          <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
            Distance
          </div>
        </div>
      </div>
    </div>
  );
}
