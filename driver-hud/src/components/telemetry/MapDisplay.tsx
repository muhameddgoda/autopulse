import { THEME } from '../../types';

interface MapDisplayProps {
  lat: number;
  lng: number;
  heading: number;
}

export function MapDisplay({ lat, lng, heading }: MapDisplayProps) {
  const mapUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${lng - 0.006}%2C${lat - 0.005}%2C${lng + 0.006}%2C${lat + 0.005}&layer=mapnik`;

  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-500 mb-1 tracking-[0.2em] uppercase">Map</div>
      <div
        className="relative overflow-hidden rounded-full"
        style={{
          width: 200,
          height: 200,
          border: `3px solid ${THEME.orange}`,
          boxShadow: `0 0 20px ${THEME.orangeGlow}`,
        }}
      >
        <iframe
          src={mapUrl}
          width="300"
          height="300"
          style={{
            border: 0,
            marginLeft: -50,
            marginTop: -50,
            filter: 'grayscale(100%) invert(92%) contrast(1.1) brightness(0.9)',
            pointerEvents: 'none',
          }}
          title="Vehicle Location"
        />

        {/* Direction Arrow */}
        <div
          className="absolute inset-0 flex items-center justify-center pointer-events-none transition-transform duration-300"
          style={{ transform: `rotate(${heading}deg)` }}
        >
          <svg width="60" height="60" viewBox="0 0 60 60">
            <defs>
              <filter id="arrowGlow" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feFlood floodColor={THEME.orange} floodOpacity="0.6" />
                <feComposite in2="blur" operator="in" />
                <feMerge>
                  <feMergeNode />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <g filter="url(#arrowGlow)">
              <path
                d="M30 6 L42 50 L30 40 L18 50 Z"
                fill={THEME.orange}
                stroke="white"
                strokeWidth="1.5"
              />
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}
