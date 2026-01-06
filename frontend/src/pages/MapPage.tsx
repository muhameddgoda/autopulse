import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import { useTelemetry } from '../hooks/useTelemetry';
import { vehicleApi } from '../lib/api';
import { Vehicle } from '../types';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Theme colors
const ORANGE = '#f97316';
const DARK_BG = '#0a0a0a';
const DARK_CARD = '#141414';
const DARK_BORDER = '#262626';
const DARK_TEXT = '#ffffff';
const DARK_TEXT_MUTED = '#737373';

// Custom marker icon
const customIcon = new L.DivIcon({
  className: 'custom-marker',
  html: `
    <div style="
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
    ">
      <svg width="36" height="36" viewBox="0 0 36 36">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="blur"/>
            <feFlood flood-color="${ORANGE}" flood-opacity="0.5"/>
            <feComposite in2="blur" operator="in"/>
            <feMerge>
              <feMergeNode/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <g filter="url(#glow)">
          <circle cx="18" cy="18" r="10" fill="${ORANGE}" stroke="white" stroke-width="2"/>
          <circle cx="18" cy="18" r="4" fill="white"/>
        </g>
      </svg>
    </div>
  `,
  iconSize: [40, 40],
  iconAnchor: [20, 20],
});

// Map center updater
function MapUpdater({ lat, lng }: { lat: number; lng: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView([lat, lng], map.getZoom());
  }, [lat, lng, map]);
  return null;
}

export default function MapPage() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);

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

  const { telemetry, connectionStatus } = useTelemetry({
    vehicleId: vehicle?.id ?? null,
    enabled: !!vehicle,
  });

  const lat = telemetry?.latitude ?? 48.8342;
  const lng = telemetry?.longitude ?? 9.1519;
  const speed = telemetry?.speed_kmh ?? 0;

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center" style={{ backgroundColor: DARK_BG }}>
        <div style={{ color: DARK_TEXT_MUTED }}>Loading...</div>
      </div>
    );
  }

  return (
    <div className="h-full p-6" style={{ backgroundColor: DARK_BG }}>
      <div 
        className="h-full rounded-2xl overflow-hidden relative"
        style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
      >
        {/* Header overlay */}
        <div className="absolute top-4 left-4 z-[1000]">
          <div 
            className="backdrop-blur-sm rounded-xl px-4 py-3 shadow-lg"
            style={{ backgroundColor: 'rgba(20, 20, 20, 0.9)', border: `1px solid ${DARK_BORDER}` }}
          >
            <h1 className="text-lg font-bold" style={{ color: DARK_TEXT }}>Live Location</h1>
            <p className="text-xs" style={{ color: DARK_TEXT_MUTED }}>{lat.toFixed(4)}, {lng.toFixed(4)}</p>
          </div>
        </div>

        {/* Speed overlay */}
        <div className="absolute top-4 right-4 z-[1000]">
          <div 
            className="backdrop-blur-sm rounded-xl px-4 py-3 shadow-lg text-center"
            style={{ backgroundColor: 'rgba(20, 20, 20, 0.9)', border: `1px solid ${DARK_BORDER}` }}
          >
            <div className="text-3xl font-bold" style={{ color: ORANGE, fontFamily: 'monospace' }}>
              {Math.round(speed)}
            </div>
            <div className="text-xs" style={{ color: DARK_TEXT_MUTED }}>km/h</div>
          </div>
        </div>

        {/* Connection status */}
        <div 
          className="absolute bottom-4 left-4 z-[1000] flex items-center gap-2 backdrop-blur-sm rounded-full px-3 py-1.5 shadow-sm"
          style={{ backgroundColor: 'rgba(20, 20, 20, 0.9)', border: `1px solid ${DARK_BORDER}` }}
        >
          <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
            {connectionStatus === 'connected' ? 'Live' : 'Offline'}
          </span>
        </div>

        {/* Map - Dark theme tiles */}
        <MapContainer
          center={[lat, lng]}
          zoom={15}
          className="h-full w-full"
          zoomControl={false}
        >
          {/* Dark theme map tiles */}
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <Marker position={[lat, lng]} icon={customIcon}>
            <Popup>
              <div className="text-center" style={{ color: '#333' }}>
                <strong>Porsche 911 Turbo S</strong><br />
                {Math.round(speed)} km/h
              </div>
            </Popup>
          </Marker>
          <MapUpdater lat={lat} lng={lng} />
        </MapContainer>
      </div>
    </div>
  );
}
