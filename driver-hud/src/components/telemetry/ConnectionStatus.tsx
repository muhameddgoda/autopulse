interface ConnectionStatusProps {
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
}

export function ConnectionStatus({ status }: ConnectionStatusProps) {
  const statusConfig = {
    connecting: { color: 'bg-yellow-500', label: 'Connecting', pulse: true },
    connected: { color: 'bg-green-500', label: 'Live', pulse: false },
    disconnected: { color: 'bg-red-500', label: 'Offline', pulse: false },
    error: { color: 'bg-red-500', label: 'Error', pulse: true },
  };

  const config = statusConfig[status];

  return (
    <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-gray-900/50">
      <div 
        className={`w-2 h-2 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`} 
      />
      <span className="text-[10px] text-gray-400 uppercase tracking-wider">
        {config.label}
      </span>
    </div>
  );
}
