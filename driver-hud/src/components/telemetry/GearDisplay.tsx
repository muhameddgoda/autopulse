import { THEME, getGearDisplay } from '../../types';

interface GearDisplayProps {
  gear: number;
}

export function GearDisplay({ gear }: GearDisplayProps) {
  const gearStr = getGearDisplay(gear);
  const isReverse = gear === -1;
  const color = isReverse ? THEME.purple : THEME.orange;

  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-500 mb-1 tracking-[0.2em] uppercase">Gear</div>
      <div
        className="w-[90px] h-[90px] rounded-full flex items-center justify-center transition-all duration-300"
        style={{
          border: `3px solid ${color}`,
          boxShadow: `0 0 15px ${color}40`,
        }}
      >
        <span
          className="text-5xl font-bold font-mono transition-colors duration-300"
          style={{ color }}
        >
          {gearStr}
        </span>
      </div>
      {/* Gear indicator dots */}
      <div className="flex gap-1 mt-2">
        {['R', 'N', '1', '2', '3', '4', '5', '6', '7'].map((g) => {
          const gearNum = g === 'R' ? -1 : g === 'N' ? 0 : parseInt(g);
          const isActive = gear === gearNum;
          const dotColor = gearNum === -1 ? THEME.purple : THEME.orange;
          return (
            <div
              key={g}
              className="w-2 h-2 rounded-full transition-all duration-200"
              style={{
                backgroundColor: isActive ? dotColor : '#333',
                boxShadow: isActive ? `0 0 8px ${dotColor}` : 'none',
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
