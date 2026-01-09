import {
  DARK_BORDER,
  DARK_TEXT,
  DARK_TEXT_MUTED,
  ORANGE,
} from "../../constants/theme";

interface SpeedGearOverlayProps {
  speed: number;
  gear: number;
}

export function SpeedGearOverlay({ speed, gear }: SpeedGearOverlayProps) {
  const gearStr = gear === -1 ? "R" : gear === 0 ? "N" : gear.toString();

  return (
    <div
      className="absolute bottom-4 left-4 backdrop-blur-sm rounded-2xl px-5 py-3 shadow-lg"
      style={{
        backgroundColor: "rgba(20, 20, 20, 0.9)",
        border: `1px solid ${DARK_BORDER}`,
      }}
    >
      <div className="flex items-end gap-4">
        <div>
          <div
            className="text-xs uppercase tracking-wider"
            style={{ color: DARK_TEXT_MUTED }}
          >
            Speed
          </div>
          <div
            className="text-4xl font-bold"
            style={{ color: DARK_TEXT, fontFamily: "monospace" }}
          >
            {Math.round(speed)}
            <span className="text-lg ml-1" style={{ color: DARK_TEXT_MUTED }}>
              km/h
            </span>
          </div>
        </div>
        <div
          className="pl-4"
          style={{ borderLeft: `1px solid ${DARK_BORDER}` }}
        >
          <div
            className="text-xs uppercase tracking-wider"
            style={{ color: DARK_TEXT_MUTED }}
          >
            Gear
          </div>
          <div
            className="text-4xl font-bold"
            style={{ color: ORANGE, fontFamily: "monospace" }}
          >
            {gearStr}
          </div>
        </div>
      </div>
    </div>
  );
}
