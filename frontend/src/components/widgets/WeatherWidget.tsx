import { useState } from "react";
import { Cloud, Sun, CloudRain } from "lucide-react";
import {
  DARK_CARD,
  DARK_BORDER,
  DARK_TEXT,
  DARK_TEXT_MUTED,
  ORANGE,
} from "../../constants/theme";

type WeatherCondition = "sunny" | "cloudy" | "rainy";

interface WeatherData {
  temp: number;
  condition: WeatherCondition;
  city: string;
}

export function WeatherWidget() {
  const [weather] = useState<WeatherData>({
    temp: 22,
    condition: "sunny",
    city: "Stuttgart",
  });

  const WeatherIcon =
    weather.condition === "sunny"
      ? Sun
      : weather.condition === "rainy"
      ? CloudRain
      : Cloud;

  return (
    <div
      className="rounded-2xl p-5 h-full"
      style={{ backgroundColor: DARK_CARD, border: `1px solid ${DARK_BORDER}` }}
    >
      <div
        className="text-xs uppercase tracking-wider mb-3"
        style={{ color: DARK_TEXT_MUTED }}
      >
        Weather
      </div>
      <div className="flex items-center gap-4">
        <WeatherIcon className="w-12 h-12" style={{ color: ORANGE }} />
        <div>
          <div className="text-3xl font-bold" style={{ color: DARK_TEXT }}>
            {weather.temp}°C
          </div>
          <div
            className="text-sm capitalize"
            style={{ color: DARK_TEXT_MUTED }}
          >
            {weather.condition}
          </div>
        </div>
      </div>
      <div className="text-xs mt-3" style={{ color: DARK_TEXT_MUTED }}>
        {weather.city}
      </div>
    </div>
  );
}
