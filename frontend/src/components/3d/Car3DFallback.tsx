import { DARK_CARD, DARK_TEXT, DARK_TEXT_MUTED } from "../../constants/theme";

// Fallback when WebGL is not available
export function Car3DFallback() {
  return (
    <div
      className="w-full h-full flex items-center justify-center"
      style={{ backgroundColor: DARK_CARD }}
    >
      <div className="text-center p-8">
        <div className="text-6xl mb-4">🚗</div>
        <h3 className="text-xl font-semibold mb-2" style={{ color: DARK_TEXT }}>
          Porsche 911 Turbo S
        </h3>
        <p className="text-sm" style={{ color: DARK_TEXT_MUTED }}>
          3D view requires WebGL support
        </p>
      </div>
    </div>
  );
}

// Check if WebGL is available
export function isWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
    );
  } catch (e) {
    return false;
  }
}
