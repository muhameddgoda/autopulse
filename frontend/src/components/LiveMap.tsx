import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default marker icons
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

L.Marker.prototype.options.icon = DefaultIcon;

// Types
interface Position {
  lat: number;
  lon: number;
  timestamp?: string;
}

interface TelemetryReading {
  latitude: number;
  longitude: number;
  speed: number;
  heading: number;
  mode: string;
  timestamp: string;
}

interface LiveMapProps {
  telemetry: TelemetryReading | null;
  showTrail?: boolean;
  trailLength?: number;
  className?: string;
}

// Mode colors for trail
const MODE_COLORS: Record<string, string> = {
  parked: '#6b7280',
  reverse: '#a855f7',
  city: '#06b6d4',
  highway: '#3b82f6',
  sport: '#f97316',
};

// Custom car marker icon
const createCarIcon = (heading: number, mode: string) => {
  const color = MODE_COLORS[mode] || '#06b6d4';
  
  return L.divIcon({
    className: 'custom-car-marker',
    html: `
      <div style="
        width: 30px;
        height: 30px;
        transform: rotate(${heading}deg);
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <svg viewBox="0 0 24 24" fill="${color}" width="24" height="24">
          <path d="M12 2L4 12l8 10 8-10L12 2z"/>
        </svg>
      </div>
    `,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });
};

// Component to recenter map when position changes
const MapRecenter: React.FC<{ position: [number, number] }> = ({ position }) => {
  const map = useMap();
  
  useEffect(() => {
    map.setView(position, map.getZoom(), { animate: true });
  }, [map, position]);
  
  return null;
};

const LiveMap: React.FC<LiveMapProps> = ({
  telemetry,
  showTrail = true,
  trailLength = 50,
  className = '',
}) => {
  const [trail, setTrail] = useState<Position[]>([]);
  const lastPositionRef = useRef<{ lat: number; lon: number } | null>(null);

  // Update trail when telemetry changes
  useEffect(() => {
    if (!telemetry) return;

    const { latitude, longitude, timestamp } = telemetry;

    // Only add if position changed significantly (> 5 meters)
    if (lastPositionRef.current) {
      const dist = calculateDistance(
        lastPositionRef.current.lat,
        lastPositionRef.current.lon,
        latitude,
        longitude
      );
      if (dist < 0.005) return; // Less than 5 meters
    }

    lastPositionRef.current = { lat: latitude, lon: longitude };

    setTrail((prev) => {
      const newTrail = [...prev, { lat: latitude, lon: longitude, timestamp }];
      return newTrail.slice(-trailLength);
    });
  }, [telemetry, trailLength]);

  // Calculate distance between two points (in km)
  const calculateDistance = (
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number => {
    const R = 6371; // Earth's radius in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  // Default position (Berlin)
  const defaultPosition: [number, number] = [52.52, 13.405];
  
  const currentPosition: [number, number] = telemetry
    ? [telemetry.latitude, telemetry.longitude]
    : defaultPosition;

  const trailPositions: [number, number][] = trail.map((p) => [p.lat, p.lon]);

  return (
    <div className={`relative ${className}`}>
      <MapContainer
        center={currentPosition}
        zoom={15}
        style={{ height: '100%', width: '100%', borderRadius: '0.75rem' }}
        zoomControl={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Trail */}
        {showTrail && trailPositions.length > 1 && (
          <Polyline
            positions={trailPositions}
            color={MODE_COLORS[telemetry?.mode ?? 'city']}
            weight={4}
            opacity={0.7}
          />
        )}

        {/* Current Position Marker */}
        {telemetry && (
          <Marker
            position={currentPosition}
            icon={createCarIcon(telemetry.heading, telemetry.mode)}
          >
            <Popup>
              <div className="text-center">
                <div className="font-bold text-lg">
                  {telemetry.speed.toFixed(0)} km/h
                </div>
                <div className="text-sm text-gray-500 capitalize">
                  {telemetry.mode} mode
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {telemetry.latitude.toFixed(5)}, {telemetry.longitude.toFixed(5)}
                </div>
              </div>
            </Popup>
          </Marker>
        )}

        {/* Recenter map on position change */}
        <MapRecenter position={currentPosition} />
      </MapContainer>

      {/* Speed Overlay */}
      {telemetry && (
        <div
          className="absolute bottom-4 left-4 px-4 py-2 rounded-lg shadow-lg"
          style={{
            backgroundColor: `${MODE_COLORS[telemetry.mode]}dd`,
            backdropFilter: 'blur(4px)',
          }}
        >
          <div className="text-white font-bold text-2xl">
            {telemetry.speed.toFixed(0)}
            <span className="text-sm ml-1">km/h</span>
          </div>
          <div className="text-white/80 text-xs capitalize">
            {telemetry.mode} mode
          </div>
        </div>
      )}

      {/* No Data Overlay */}
      {!telemetry && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-xl">
          <div className="text-white text-center">
            <p className="text-lg font-semibold">No GPS Data</p>
            <p className="text-sm text-gray-300">Waiting for telemetry...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveMap;