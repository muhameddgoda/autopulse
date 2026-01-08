import { THEME } from '../../types';

interface AlertIconProps {
  type: 'engine' | 'oil' | 'fuel' | 'battery' | 'tire' | 'rpm';
  active: boolean;
  critical?: boolean;
}

const iconPaths: Record<string, string> = {
  engine: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z',
  oil: 'M19 14V6c0-1.1-.9-2-2-2H7c-1.1 0-2 .9-2 2v8c0 2.21 1.79 4 4 4h1v2H8v2h8v-2h-2v-2h1c2.21 0 4-1.79 4-4zM8 8h8v2H8V8z',
  fuel: 'M19.77 7.23l.01-.01-3.72-3.72L15 4.56l2.11 2.11c-.94.36-1.61 1.26-1.61 2.33 0 1.38 1.12 2.5 2.5 2.5.36 0 .69-.08 1-.21v7.21c0 .55-.45 1-1 1s-1-.45-1-1V14c0-1.1-.9-2-2-2h-1V5c0-1.1-.9-2-2-2H6c-1.1 0-2 .9-2 2v16h10v-7.5h1.5v5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V9c0-.69-.28-1.32-.73-1.77zM12 10H6V5h6v5z',
  battery: 'M15.67 4H14V2h-4v2H8.33C7.6 4 7 4.6 7 5.33v15.33C7 21.4 7.6 22 8.33 22h7.33c.74 0 1.34-.6 1.34-1.33V5.33C17 4.6 16.4 4 15.67 4z',
  tire: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6z',
  rpm: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z',
};

export function AlertIcon({ type, active, critical = false }: AlertIconProps) {
  if (!active) return null;

  const color = critical ? THEME.red : THEME.yellow;

  return (
    <div
      className={`${critical ? 'animate-pulse' : ''}`}
      title={`${type.toUpperCase()} ${critical ? 'CRITICAL' : 'WARNING'}`}
    >
      <svg 
        className="w-5 h-5" 
        viewBox="0 0 24 24" 
        fill={color}
        style={{ filter: `drop-shadow(0 0 4px ${color})` }}
      >
        <path d={iconPaths[type]} />
      </svg>
    </div>
  );
}
