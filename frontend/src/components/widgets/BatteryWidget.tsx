import {
  DARK_CARD,
  DARK_BORDER,
  DARK_TEXT,
  DARK_TEXT_MUTED,
  ORANGE,
} from "../../constants/theme";

interface BatteryWidgetProps {
  voltage?: number;
  speed?: number;
}

export function BatteryWidget({
  voltage = 12.6,
  speed = 0,
}: BatteryWidgetProps) {
  const safeVoltage = voltage ?? 12.6;
  const safeSpeed = speed ?? 0;
  const percentage = Math.min(
    100,
    Math.max(0, ((safeVoltage - 11.5) / 3) * 100)
  );
  const isCharging = safeVoltage > 13.5 && safeSpeed === 0;
  const fillColor = percentage > 20 ? ORANGE : "#ef4444";

  return (
    <div
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div
        className="text-xs uppercase tracking-wider mb-3"
        style={{ color: DARK_TEXT_MUTED }}
      >
        Battery
      </div>
      <div className="flex items-center gap-4">
        {/* SVG Battery Icon */}
        <svg width="56" height="32" viewBox="0 0 56 32">
          {/* Battery body outline */}
          <rect
            x="2"
            y="4"
            width="44"
            height="24"
            rx="4"
            ry="4"
            fill="none"
            stroke={DARK_TEXT_MUTED}
            strokeWidth="2"
          />
          {/* Battery tip */}
          <rect
            x="46"
            y="10"
            width="6"
            height="12"
            rx="2"
            ry="2"
            fill={DARK_TEXT_MUTED}
          />
          {/* Battery fill */}
          <rect
            x="6"
            y="8"
            width={Math.max(0, (percentage / 100) * 36)}
            height="16"
            rx="2"
            ry="2"
            fill={fillColor}
          />
        </svg>
        <div>
          <div className="text-3xl font-bold" style={{ color: DARK_TEXT }}>
            {percentage.toFixed(0)}%
          </div>
          <div className="text-sm" style={{ color: DARK_TEXT_MUTED }}>
            {isCharging ? "Charging" : "In Use"}
          </div>
        </div>
      </div>
      <div className="text-xs mt-3" style={{ color: DARK_TEXT_MUTED }}>
        {safeVoltage.toFixed(1)}V
      </div>
    </div>
  );
}
