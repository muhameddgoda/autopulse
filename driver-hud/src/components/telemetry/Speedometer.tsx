import { useEffect, useRef } from "react";
import { useSmoothValue } from "../../hooks/useSmoothValue";
import { THEME } from "../../types";

interface SpeedometerProps {
  speed: number;
  maxSpeed?: number;
}

export function Speedometer({ speed, maxSpeed = 320 }: SpeedometerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothSpeed = useSmoothValue(speed, 0.12);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = canvas.width;
    const center = size / 2;
    const radius = size / 2 - 30;

    ctx.clearRect(0, 0, size, size);

    // Arc angles
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const angleRange = endAngle - startAngle;

    // Background arc
    ctx.beginPath();
    ctx.arc(center, center, radius, startAngle, endAngle);
    ctx.strokeStyle = THEME.orangeDim;
    ctx.lineWidth = 4;
    ctx.stroke();

    // Speed markers
    const markers = [0, 40, 80, 120, 160, 200, 240, 280, 320];
    markers.forEach((val) => {
      const angle = startAngle + (val / maxSpeed) * angleRange;
      const innerR = radius - 15;
      const outerR = radius - 5;

      ctx.beginPath();
      ctx.moveTo(
        center + Math.cos(angle) * innerR,
        center + Math.sin(angle) * innerR
      );
      ctx.lineTo(
        center + Math.cos(angle) * outerR,
        center + Math.sin(angle) * outerR
      );
      ctx.strokeStyle = THEME.orange;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Labels
      const textR = radius + 15;
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      ctx.fillStyle = THEME.orange;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        val.toString(),
        center + Math.cos(angle) * textR,
        center + Math.sin(angle) * textR
      );
    });

    // Minor ticks
    for (let val = 0; val <= maxSpeed; val += 20) {
      if (markers.includes(val)) continue;
      const angle = startAngle + (val / maxSpeed) * angleRange;
      const innerR = radius - 10;
      const outerR = radius - 5;

      ctx.beginPath();
      ctx.moveTo(
        center + Math.cos(angle) * innerR,
        center + Math.sin(angle) * innerR
      );
      ctx.lineTo(
        center + Math.cos(angle) * outerR,
        center + Math.sin(angle) * outerR
      );
      ctx.strokeStyle = THEME.orangeDim;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Needle
    const needleAngle = startAngle + (smoothSpeed / maxSpeed) * angleRange;
    const needleLength = radius - 35;

    // Needle glow
    ctx.save();
    ctx.shadowColor = THEME.orange;
    ctx.shadowBlur = 15;

    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.lineTo(
      center + Math.cos(needleAngle) * needleLength,
      center + Math.sin(needleAngle) * needleLength
    );
    ctx.strokeStyle = THEME.orange;
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.stroke();
    ctx.restore();

    // Center cap
    ctx.beginPath();
    ctx.arc(center, center, 12, 0, Math.PI * 2);
    ctx.fillStyle = THEME.orange;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(center, center, 6, 0, Math.PI * 2);
    ctx.fillStyle = THEME.background;
    ctx.fill();
  }, [smoothSpeed, maxSpeed]);

  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] text-gray-500 mb-1 tracking-[0.2em] uppercase">
        Speed
      </div>
      <div className="relative">
        <canvas ref={canvasRef} width={300} height={280} />
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-10">
          <span className="text-sm text-gray-500 tracking-wider pb-20">km/h</span>
          <span
            className="text-6xl font-bold font-mono tabular-nums"
            style={{
              color: THEME.orange,
              textShadow: `0 0 30px ${THEME.orangeGlow}`,
            }}
          >
            {Math.round(smoothSpeed)}
          </span>
        </div>
      </div>
    </div>
  );
}
