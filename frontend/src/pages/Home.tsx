import { useState, useEffect, Suspense, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Environment,
  ContactShadows,
  PerspectiveCamera,
} from "@react-three/drei";

import { useTelemetry } from "../hooks/useTelemetry";
import { vehicleApi } from "../lib/api";
import { Vehicle } from "../types";
import {
  DARK_BG,
  DARK_CARD,
  DARK_BORDER,
  DARK_TEXT,
  DARK_TEXT_MUTED,
} from "../constants/theme";
import {
  PorscheModel,
  ModelLoader,
  Car3DFallback,
  isWebGLAvailable,
} from "../components/3d";
import {
  SpeedGearOverlay,
  WeatherWidget,
  BatteryWidget,
  MusicWidget,
  SpeedGraphWidget,
  TripInfoWidget,
} from "../components/widgets";
import WebGLErrorBoundary from "../components/WebGLErrorBoundary";

export default function Home() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);
  const [speedHistory, setSpeedHistory] = useState<
    { speed: number; time: string }[]
  >([]);
  const [webGLAvailable] = useState(() => isWebGLAvailable());
  const historyRef = useRef<{ speed: number; time: string }[]>([]);

  // Handle trip end - clear speed history
  const handleTripEnd = () => {
    historyRef.current = [];
    setSpeedHistory([]);
  };

  useEffect(() => {
    async function fetchVehicle() {
      try {
        const vehicles = await vehicleApi.getAll();
        if (vehicles.length > 0) setVehicle(vehicles[0]);
      } catch (error) {
        console.error("Failed to fetch vehicle:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchVehicle();
  }, []);

  const { telemetry, connectionStatus } = useTelemetry({
    vehicleId: vehicle?.id ?? null,
    enabled: !!vehicle,
  });

  useEffect(() => {
    if (telemetry) {
      const now = new Date().toLocaleTimeString();
      const newPoint = { speed: telemetry.speed_kmh ?? 0, time: now };

      if (historyRef.current.length >= 60) {
        historyRef.current.shift();
      }
      historyRef.current.push(newPoint);
      setSpeedHistory([...historyRef.current]);
    }
  }, [telemetry]);

  if (loading) {
    return (
      <div
        className="h-full flex items-center justify-center"
        style={{ backgroundColor: DARK_BG }}
      >
        <div style={{ color: DARK_TEXT_MUTED }}>Loading...</div>
      </div>
    );
  }

  const data = telemetry ?? {
    speed_kmh: 0,
    rpm: 0,
    gear: 0,
    battery_voltage: 12.6,
  };

  return (
    <div
      className="h-full p-6 overflow-auto"
      style={{ backgroundColor: DARK_BG }}
    >
      {/* Grid Layout */}
      <div className="grid grid-cols-12 gap-5 h-full">
        {/* Left Column - 3D Car + Speed History underneath */}
        <div className="col-span-7 flex flex-col gap-5">
          {/* 3D Car */}
          <div
            className="rounded-2xl relative overflow-hidden flex-1"
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
              minHeight: "400px",
            }}
          >
            {/* Connection Status */}
            <div
              className="absolute top-4 right-4 z-10 flex items-center gap-2 backdrop-blur-sm rounded-full px-3 py-1.5"
              style={{ backgroundColor: "rgba(20, 20, 20, 0.8)" }}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  connectionStatus === "connected"
                    ? "bg-green-500"
                    : "bg-red-500"
                }`}
              />
              <span className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
                {connectionStatus === "connected" ? "Live" : "Offline"}
              </span>
            </div>

            {/* 3D Canvas with Error Boundary */}
            {webGLAvailable ? (
              <WebGLErrorBoundary fallback={<Car3DFallback />}>
                <Canvas shadows>
                  <PerspectiveCamera
                    makeDefault
                    position={[6, 2, 7]}
                    fov={40}
                  />
                  <OrbitControls
                    enablePan={false}
                    enableZoom={true}
                    minDistance={5}
                    maxDistance={12}
                    maxPolarAngle={Math.PI / 2.1}
                    minPolarAngle={Math.PI / 6}
                    autoRotate
                    autoRotateSpeed={0.5}
                  />

                  <Environment preset="studio" />

                  <Suspense fallback={<ModelLoader />}>
                    <PorscheModel />
                  </Suspense>

                  {/* Adjusted ground position to not cut car */}
                  <ContactShadows
                    position={[0, -1.2, 0]}
                    opacity={0.5}
                    scale={12}
                    blur={2}
                    far={10}
                  />
                </Canvas>
              </WebGLErrorBoundary>
            ) : (
              <Car3DFallback />
            )}

            {/* Speed & Gear Overlay */}
            <SpeedGearOverlay speed={data.speed_kmh} gear={data.gear} />

            {/* Car Name */}
            <div className="absolute top-4 left-4">
              <div className="text-lg font-bold" style={{ color: DARK_TEXT }}>
                Porsche 911
              </div>
              <div className="text-sm" style={{ color: DARK_TEXT_MUTED }}>
                Turbo S • 2024
              </div>
            </div>
          </div>

          {/* Speed History - Full width under 3D model */}
          <SpeedGraphWidget history={speedHistory} />
        </div>

        {/* Right Column */}
        <div className="col-span-5 flex flex-col gap-5">
          {/* Top Row - Weather & Battery */}
          <div className="grid grid-cols-2 gap-5">
            <WeatherWidget />
            <BatteryWidget
              voltage={data.battery_voltage}
              speed={data.speed_kmh}
            />
          </div>

          {/* Music Player */}
          <MusicWidget />

          {/* Current Trip */}
          <TripInfoWidget
            vehicleId={vehicle?.id ?? null}
            onTripEnd={handleTripEnd}
          />
        </div>
      </div>
    </div>
  );
}
