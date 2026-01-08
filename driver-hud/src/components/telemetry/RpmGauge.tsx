import { useEffect, useRef } from 'react';
import { useSmoothValue } from '../../hooks/useSmoothValue';
import { THEME, THRESHOLDS } from '../../types';

interface RpmGaugeProps {
  rpm: number;
  maxRpm?: number;
}

export function RpmGauge({ rpm, maxRpm = 8000 }: RpmGaugeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothRpm = useSmoothValue(rpm, 0.15);
  const isHigh = rpm >= THRESHOLDS.rpm.warning;
  const isRedline = rpm >= THRESHOLDS.rpm.redline;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width;
    const center = size / 2;
    const radius = size / 2 - 20;

    ctx.clearRect(0, 0, size, size);

    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const angleRange = endAngle - startAngle;

    // Background arc
    ctx.beginPath();
    ctx.arc(center, center, radius, startAngle, endAngle);
    ctx.strokeStyle = THEME.orangeDim;
    ctx.lineWidth = 3;
    ctx.stroke();

    // Redline zone (7000-8000)
    const redlineStart = startAngle + (7000 / maxRpm) * angleRange;
    ctx.beginPath();
    ctx.arc(center, center, radius, redlineStart, endAngle);
    ctx.strokeStyle = THEME.redDim;
    ctx.lineWidth = 3;
    ctx.stroke();

    // RPM markers
    const markers = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    markers.forEach((val) => {
      const angle = startAngle + (val / 8) * angleRange;
      const isRedlineMarker = val >= 7;

      const innerR = radius - 12;
      const outerR = radius - 2;
      ctx.beginPath();
      ctx.moveTo(
        center + Math.cos(angle) * innerR,
        center + Math.sin(angle) * innerR
      );
      ctx.lineTo(
        center + Math.cos(angle) * outerR,
        center + Math.sin(angle) * outerR
      );
      ctx.strokeStyle = isRedlineMarker ? THEME.red : THEME.orange;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Labels
      const textR = radius + 12;
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.fillStyle = isRedlineMarker ? THEME.red : THEME.orange;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        val.toString(),
        center + Math.cos(angle) * textR,
        center + Math.sin(angle) * textR
      );
    });

    // Needle
    const needleAngle = startAngle + (smoothRpm / maxRpm) * angleRange;
    const needleLength = radius - 25;

    ctx.save();
    ctx.shadowColor = isRedline ? THEME.red : THEME.orange;
    ctx.shadowBlur = 10;

    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.lineTo(
      center + Math.cos(needleAngle) * needleLength,
      center + Math.sin(needleAngle) * needleLength
    );
    ctx.strokeStyle = isRedline ? THEME.red : THEME.orange;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.restore();

    // Center dot
    ctx.beginPath();
    ctx.arc(center, center, 6, 0, Math.PI * 2);
    ctx.fillStyle = isRedline ? THEME.red : THEME.orange;
    ctx.fill();
  }, [smoothRpm, maxRpm, isRedline]);

  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-500 mb-1 tracking-[0.2em] uppercase">RPM</div>
      <div className="relative">
        <canvas ref={canvasRef} width={160} height={160} />
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-4">
          <span className="text-[10px] text-gray-600 pb-10">RPM</span>
          <span
            className="text-2xl font-bold font-mono tabular-nums"
            style={{ color: isRedline ? THEME.red : THEME.orange }}
          >
            {Math.round(smoothRpm)}
          </span>
        </div>
      </div>
    </div>
  );
}
