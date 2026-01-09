import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";
import {
  DARK_CARD,
  DARK_BORDER,
  DARK_TEXT_MUTED,
  ORANGE,
} from "../../constants/theme";

interface SpeedGraphWidgetProps {
  history: { speed: number; time: string }[];
}

export function SpeedGraphWidget({ history }: SpeedGraphWidgetProps) {
  return (
    <div
      className="rounded-2xl p-5"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div
        className="text-xs uppercase tracking-wider mb-3"
        style={{ color: DARK_TEXT_MUTED }}
      >
        Speed History
      </div>
      <div className="h-24">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history.length > 0 ? history : [{ speed: 0 }]}>
            <YAxis hide domain={[0, "auto"]} />
            <Line
              type="monotone"
              dataKey="speed"
              stroke={ORANGE}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div
        className="flex justify-between text-xs mt-1"
        style={{ color: DARK_TEXT_MUTED }}
      >
        <span>60s ago</span>
        <span>Now</span>
      </div>
    </div>
  );
}
