import { useEffect, useRef } from 'react';
import { useSmoothValue } from '../../hooks/useSmoothValue';
import { THEME, THRESHOLDS } from '../../types';

interface FuelGaugeProps {
  level: number;
}

export function FuelGauge({ level }: FuelGaugeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothLevel = useSmoothValue(level, 0.1);
  const isLow = level <= THRESHOLDS.fuel_level.warning;
  const isCritical = level <= THRESHOLDS.fuel_level.critical;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width;
    const center = size / 2;
    const radius = size / 2 - 12;

    ctx.clearRect(0, 0, size, size);

    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const angleRange = endAngle - startAngle;

    // Background arc
    ctx.beginPath();
    ctx.arc(center, center, radius, startAngle, endAngle);
    ctx.strokeStyle = THEME.orangeDim;
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Fuel level arc
    const fuelAngle = startAngle + (smoothLevel / 100) * angleRange;
    ctx.beginPath();
    ctx.arc(center, center, radius, startAngle, fuelAngle);
    ctx.strokeStyle = isCritical ? THEME.red : isLow ? THEME.yellow : THEME.orange;
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.stroke();

    // E and F labels
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'center';
    const labelR = radius - 15;
    ctx.fillText(
      'E',
      center + Math.cos(startAngle) * labelR,
      center + Math.sin(startAngle) * labelR + 15
    );
    ctx.fillText(
      'F',
      center + Math.cos(endAngle) * labelR,
      center + Math.sin(endAngle) * labelR + 15
    );
  }, [smoothLevel, isLow, isCritical]);

  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-500 mb-1 tracking-[0.2em] uppercase">Fuel</div>
      <div className="relative">
        <canvas ref={canvasRef} width={90} height={90} />
        <div className="absolute inset-0 flex items-center justify-center">
          <span
            className="text-lg font-bold font-mono"
            style={{ color: isCritical ? THEME.red : isLow ? THEME.yellow : THEME.orange }}
          >
            {Math.round(smoothLevel)}%
          </span>
        </div>
      </div>
    </div>
  );
}
