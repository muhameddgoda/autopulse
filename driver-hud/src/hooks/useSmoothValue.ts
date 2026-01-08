import { useState, useEffect, useRef } from 'react';

/**
 * Hook that smoothly animates a value towards a target
 * Great for gauge needles and numeric displays
 */
export function useSmoothValue(target: number, smoothing: number = 0.1): number {
  const [value, setValue] = useState(target);
  const animationRef = useRef<number>();

  useEffect(() => {
    const animate = () => {
      setValue(prev => {
        const diff = target - prev;
        if (Math.abs(diff) < 0.5) return target;
        return prev + diff * smoothing;
      });
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [target, smoothing]);

  return value;
}
