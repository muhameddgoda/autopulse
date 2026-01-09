import { useState } from "react";
import {
  Music,
  Pause,
  Play,
  SkipForward,
  SkipBack,
  Smartphone,
} from "lucide-react";
import {
  DARK_CARD,
  DARK_BORDER,
  DARK_TEXT,
  DARK_TEXT_MUTED,
  ORANGE,
  ORANGE_DIM,
} from "../../constants/theme";

export function MusicWidget() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [connected, setConnected] = useState(false);

  if (!connected) {
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
          Music
        </div>
        <div className="flex flex-col items-center justify-center py-4">
          <Smartphone
            className="w-10 h-10 mb-3"
            style={{ color: DARK_TEXT_MUTED }}
          />
          <div className="text-sm mb-3" style={{ color: DARK_TEXT_MUTED }}>
            Connect your phone
          </div>
          <button
            onClick={() => setConnected(true)}
            className="px-4 py-2 rounded-full text-sm font-medium text-white transition-all hover:opacity-90"
            style={{ backgroundColor: ORANGE }}
          >
            Connect via Bluetooth
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div
        className="text-xs uppercase tracking-wider mb-3"
        style={{ color: DARK_TEXT_MUTED }}
      >
        Now Playing
      </div>
      <div className="flex items-center gap-4 mb-4">
        <div
          className="w-16 h-16 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: ORANGE_DIM }}
        >
          <Music className="w-8 h-8" style={{ color: ORANGE }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-semibold truncate" style={{ color: DARK_TEXT }}>
            Night Drive
          </div>
          <div className="text-sm truncate" style={{ color: DARK_TEXT_MUTED }}>
            Synthwave Mix
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div
        className="w-full h-1 rounded-full mb-3"
        style={{ backgroundColor: DARK_BORDER }}
      >
        <div
          className="h-full rounded-full"
          style={{ width: "45%", backgroundColor: ORANGE }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-6">
        <button className="p-2 rounded-full transition-colors hover:bg-white/10">
          <SkipBack className="w-5 h-5" style={{ color: DARK_TEXT_MUTED }} />
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="p-3 rounded-full"
          style={{ backgroundColor: ORANGE }}
        >
          {isPlaying ? (
            <Pause className="w-6 h-6" style={{ color: DARK_TEXT }} />
          ) : (
            <Play className="w-6 h-6" style={{ color: DARK_TEXT }} />
          )}
        </button>
        <button className="p-2 rounded-full transition-colors hover:bg-white/10">
          <SkipForward className="w-5 h-5" style={{ color: DARK_TEXT_MUTED }} />
        </button>
      </div>
    </div>
  );
}
