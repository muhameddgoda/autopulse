import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Custom car icon
const carIcon = new L.DivIcon({
  className: 'car-marker',
  html: `
    <div style="
      width: 40px;
      height: 40px;
      background: #D5001C;
      border: 3px solid white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    ">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
        <path d="M5 17a2 2 0 1 0 4 0 2 2 0 0 0-4 0m10 0a2 2 0 1 0 4 0 2 2 0 0 0-4 0m-3-10l-4 4h3v4h2v-4h3l-4-4z"/>
      </svg>
    </div>
  `,
  iconSize: [40, 40],
  iconAnchor: [20, 20],
});

// Component to update map view when position changes
function MapUpdater({ position }: { position: [number, number] }) {
  const map = useMap();
  
  useEffect(() => {
    map.setView(position, map.getZoom(), { animate: true, duration: 0.5 });
  }, [map, position]);
  
  return null;
}

interface LiveMapProps {
  latitude: number;
  longitude: number;
  speed: number;
  heading?: number;
  trail?: Array<[number, number]>;
}

export default function LiveMap({ latitude, longitude, speed, trail = [] }: LiveMapProps) {
  const position: [number, number] = [latitude, longitude];
  const trailRef = useRef<Array<[number, number]>>([]);
  
  // Keep last 100 positions for trail
  useEffect(() => {
    if (latitude && longitude) {
      trailRef.current = [...trailRef.current.slice(-99), [latitude, longitude]];
    }
  }, [latitude, longitude]);

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden">
      <MapContainer
        center={position}
        zoom={16}
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        attributionControl={false}
      >
        {/* Dark map tiles */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />
        
        {/* Trail line */}
        {trailRef.current.length > 1 && (
          <Polyline
            positions={trailRef.current}
            pathOptions={{
              color: '#D5001C',
              weight: 3,
              opacity: 0.7,
            }}
          />
        )}
        
        {/* Car marker */}
        <Marker position={position} icon={carIcon} />
        
        {/* Auto-follow car */}
        <MapUpdater position={position} />
      </MapContainer>
      
      {/* Speed overlay */}
      <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 z-[1000]">
        <div className="text-porsche-gray-400 text-xs">SPEED</div>
        <div className="text-white text-2xl font-bold font-mono">
          {Math.round(speed)} <span className="text-sm text-porsche-gray-400">km/h</span>
        </div>
      </div>
      
      {/* Coordinates overlay */}
      <div className="absolute bottom-4 right-4 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 z-[1000]">
        <div className="text-porsche-gray-400 text-xs font-mono">
          {latitude.toFixed(5)}, {longitude.toFixed(5)}
        </div>
      </div>
    </div>
  );
}
