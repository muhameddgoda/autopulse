import { useState, useEffect, Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  Environment,
  ContactShadows,
  useGLTF,
  Center,
  PerspectiveCamera,
  Html,
} from "@react-three/drei";
import { useTelemetry } from "../hooks/useTelemetry";
import { vehicleApi } from "../lib/api";
import { Vehicle, THRESHOLDS } from "../types";
import * as THREE from "three";

// Theme colors
const ORANGE = "#f97316";
const DARK_BG = "#0a0a0a";
const DARK_CARD = "#141414";
const DARK_BORDER = "#262626";
const DARK_TEXT = "#ffffff";
const DARK_TEXT_MUTED = "#737373";

interface TireData {
  fl: number;
  fr: number;
  rl: number;
  rr: number;
}

// Get tire status color
function getTireColor(pressure: number | undefined): string {
  const p = pressure ?? 33;
  if (p <= THRESHOLDS.tire_pressure.critical) return "#ef4444";
  if (p <= THRESHOLDS.tire_pressure.low) return "#f59e0b";
  if (p >= (THRESHOLDS.tire_pressure.high || 38)) return "#f59e0b";
  return "#22c55e";
}

// 3D Tire Pressure Label
function TireLabel({
  position,
  pressure,
  label,
}: {
  position: [number, number, number];
  pressure: number | undefined;
  label: string;
}) {
  const safePressure = pressure ?? 33;
  const color = getTireColor(safePressure);

  return (
    <Html position={position} center>
      <div
        className="flex flex-col items-center px-3 py-2 rounded-lg"
        style={{
          backgroundColor: "rgba(0,0,0,0.85)",
          border: `2px solid ${color}`,
          boxShadow: `0 0 20px ${color}60`,
          minWidth: "80px",
        }}
      >
        <span className="text-[10px] text-gray-400 uppercase">{label}</span>
        <span className="text-2xl font-bold font-mono" style={{ color }}>
          {safePressure.toFixed(1)}
        </span>
        <span className="text-[10px] text-gray-500">PSI</span>
      </div>
    </Html>
  );
}

// 3D Porsche Model with Tire Labels - STATIC TOP VIEW
function PorscheModelWithTires({ tires }: { tires: TireData }) {
  const groupRef = useRef<THREE.Group>(null);
  const { scene } = useGLTF("/models/porsche911.glb");

  // Tire positions for TOP VIEW (adjusted for bird's eye perspective)
  const tirePositions: { [key: string]: [number, number, number] } = {
    fl: [-2.5, 0.5, 1.8], // Front Left
    fr: [2.5, 0.5, 1.8], // Front Right
    rl: [-2.5, 0.5, -1.8], // Rear Left
    rr: [2.5, 0.5, -1.8], // Rear Right
  };

  return (
    <group ref={groupRef}>
      <Center>
        <primitive
          object={scene.clone()}
          scale={1.0}
          rotation={[0, Math.PI, 0]} // Car facing up
        />
      </Center>

      {/* Tire Labels */}
      <TireLabel position={tirePositions.fl} pressure={tires.fl} label="FL" />
      <TireLabel position={tirePositions.fr} pressure={tires.fr} label="FR" />
      <TireLabel position={tirePositions.rl} pressure={tires.rl} label="RL" />
      <TireLabel position={tirePositions.rr} pressure={tires.rr} label="RR" />
    </group>
  );
}

useGLTF.preload("/models/porsche911.glb");

// Loading spinner
function ModelLoader() {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y += delta * 2;
  });
  return (
    <mesh ref={meshRef}>
      <torusGeometry args={[1, 0.2, 16, 32]} />
      <meshStandardMaterial color={ORANGE} wireframe />
    </mesh>
  );
}

// Stat Item Component
function StatItem({
  label,
  value,
  unit,
  warning = false,
}: {
  label: string;
  value: string | number;
  unit?: string;
  warning?: boolean;
}) {
  return (
    <div
      className="flex items-center justify-between py-2 border-b"
      style={{ borderColor: DARK_BORDER }}
    >
      <span className="text-sm" style={{ color: DARK_TEXT_MUTED }}>
        {label}
      </span>
      <span
        className="text-sm font-mono font-semibold"
        style={{ color: warning ? "#f59e0b" : DARK_TEXT }}
      >
        {value}
        {unit && (
          <span className="text-xs ml-1" style={{ color: DARK_TEXT_MUTED }}>
            {unit}
          </span>
        )}
      </span>
    </div>
  );
}

// Main Vehicle Page
export default function VehiclePage() {
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [loading, setLoading] = useState(true);
  const [totalDistance, setTotalDistance] = useState(12847); // Simulated odometer

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

  // Update simulated odometer
  useEffect(() => {
    if (telemetry && telemetry.speed_kmh > 0) {
      const interval = setInterval(() => {
        setTotalDistance((prev) => prev + telemetry.speed_kmh / 3600);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [telemetry?.speed_kmh]);

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
    engine_temp: 20,
    oil_temp: 20,
    oil_pressure: 0,
    fuel_level: 85,
    battery_voltage: 12.6,
    tire_pressure_fl: 33,
    tire_pressure_fr: 33,
    tire_pressure_rl: 32,
    tire_pressure_rr: 32,
    speed_kmh: 0,
  };

  const tires: TireData = {
    fl: data.tire_pressure_fl,
    fr: data.tire_pressure_fr,
    rl: data.tire_pressure_rl,
    rr: data.tire_pressure_rr,
  };

  return (
    <div
      className="h-full flex flex-col overflow-hidden"
      style={{ backgroundColor: DARK_BG }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 pb-2">
        <div>
          <h1 className="text-xl font-bold" style={{ color: DARK_TEXT }}>
            Tyre Pressure
          </h1>
          <p className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
            {vehicle?.make || "Porsche"} {vehicle?.model || "911"}{" "}
            {vehicle?.variant || "Turbo S"}
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div
            className="flex items-center gap-2 rounded-full px-3 py-1.5"
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
            }}
          >
            <div
              className={`w-2 h-2 rounded-full ${
                connectionStatus === "connected" ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-xs" style={{ color: DARK_TEXT_MUTED }}>
              {connectionStatus === "connected" ? "Live" : "Offline"}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content - No Scroll */}
      <div className="flex-1 flex gap-4 p-4 pt-2 min-h-0">
        {/* Left Side - 3D Car with Tire Pressure (TOP VIEW) */}
        <div
          className="flex-1 rounded-2xl relative overflow-hidden"
          style={{
            backgroundColor: DARK_CARD,
            border: `1px solid ${DARK_BORDER}`,
          }}
        >
          {/* 3D Canvas - STATIC TOP VIEW */}
          <Canvas shadows className="!h-full">
            <PerspectiveCamera
              makeDefault
              position={[0, 10, 0.5]} // Top-down view, slightly tilted
              fov={50}
              rotation={[-Math.PI / 2, 0, 0]} // Looking down
            />
            {/* NO OrbitControls = Static view */}

            <ambientLight intensity={1.0} />
            <directionalLight position={[5, 10, 5]} intensity={1.2} />
            <directionalLight position={[-5, 10, -5]} intensity={0.8} />

            <Environment preset="studio" />

            <Suspense fallback={<ModelLoader />}>
              <PorscheModelWithTires tires={tires} />
            </Suspense>

            <ContactShadows
              position={[0, -1.1, 0]}
              opacity={0.4}
              scale={15}
              blur={2.5}
              far={10}
            />
          </Canvas>

          {/* Legend */}
          <div
            className="absolute bottom-4 left-4 flex gap-4 text-xs"
            style={{ color: DARK_TEXT_MUTED }}
          >
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span>Normal (30-37 PSI)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-amber-500" />
              <span>Warning</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span>Critical</span>
            </div>
          </div>
        </div>

        {/* Right Side - Stats Only */}
        <div className="w-64 flex flex-col gap-4">
          {/* Vehicle Stats Card */}
          <div
            className="rounded-2xl p-4 flex-1"
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
            }}
          >
            <div
              className="text-xs uppercase tracking-wider mb-3"
              style={{ color: DARK_TEXT_MUTED }}
            >
              Vehicle Stats
            </div>
            <div className="space-y-1">
              <StatItem
                label="Odometer"
                value={totalDistance.toFixed(0)}
                unit="km"
              />
              <StatItem
                label="Engine Temp"
                value={data.engine_temp.toFixed(0)}
                unit="°C"
                warning={data.engine_temp > 100}
              />
              <StatItem
                label="Oil Temp"
                value={data.oil_temp.toFixed(0)}
                unit="°C"
                warning={data.oil_temp > 120}
              />
              <StatItem
                label="Oil Pressure"
                value={(data.oil_pressure ?? 0).toFixed(1)}
                unit="bar"
                warning={(data.oil_pressure ?? 0) < 1.5}
              />
              <StatItem
                label="Fuel Level"
                value={(data.fuel_level ?? 0).toFixed(0)}
                unit="%"
                warning={(data.fuel_level ?? 0) < 15}
              />
              <StatItem
                label="Battery"
                value={(data.battery_voltage ?? 12.6).toFixed(1)}
                unit="V"
                warning={(data.battery_voltage ?? 12.6) < 12.2}
              />
            </div>
          </div>

          {/* Tire Summary Card */}
          <div
            className="rounded-2xl p-4"
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
            }}
          >
            <div
              className="text-xs uppercase tracking-wider mb-3"
              style={{ color: DARK_TEXT_MUTED }}
            >
              Tyre Pressure Summary
            </div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Front L", value: tires.fl ?? 33 },
                { label: "Front R", value: tires.fr ?? 33 },
                { label: "Rear L", value: tires.rl ?? 32 },
                { label: "Rear R", value: tires.rr ?? 32 },
              ].map((tire) => (
                <div
                  key={tire.label}
                  className="rounded-lg p-2 text-center"
                  style={{
                    backgroundColor: `${getTireColor(tire.value)}15`,
                    border: `1px solid ${getTireColor(tire.value)}40`,
                  }}
                >
                  <div
                    className="text-[10px]"
                    style={{ color: DARK_TEXT_MUTED }}
                  >
                    {tire.label}
                  </div>
                  <div
                    className="text-lg font-bold font-mono"
                    style={{ color: getTireColor(tire.value) }}
                  >
                    {(tire.value ?? 0).toFixed(1)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
