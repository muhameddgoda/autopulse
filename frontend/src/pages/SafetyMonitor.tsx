import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Eye,
  EyeOff,
  AlertTriangle,
  Camera,
  CameraOff,
  Shield,
  Volume2,
  VolumeX,
  Play,
  Square,
  Car,
  Frown,
  Focus,
  Activity,
} from "lucide-react";
import {
  DARK_BG,
  DARK_CARD,
  DARK_BORDER,
  DARK_TEXT,
  DARK_TEXT_MUTED,
  ORANGE,
} from "../constants/theme";

interface YawnMetrics {
  mar: number;
  is_yawning: boolean;
  yawn_duration_ms: number;
  yawn_count: number;
  yawns_per_minute: number;
  fatigue_level: "low" | "moderate" | "high";
}

interface DistractionMetrics {
  pitch: number;
  yaw: number;
  roll: number;
  is_distracted: boolean;
  distraction_type: string;
  distraction_duration_ms: number;
  looking_at_road: boolean;
  attention_score: number;
}

interface DrowsinessState {
  ear_average: number;
  eyes_closed: boolean;
  closed_duration_ms: number;
  blink_count: number;
  blinks_per_minute: number;
  alert_level: "none" | "warning" | "alert" | "critical";
  is_drowsy: boolean;
  face_detected: boolean;
  confidence: number;
  landmarks?: { left_eye: number[][]; right_eye: number[][] };
  yawn?: YawnMetrics;
  distraction?: DistractionMetrics;
}

interface SafetyState {
  drowsiness: DrowsinessState;
  safety_score: number;
  session_stats: {
    duration_seconds: number;
    total_drowsy_seconds: number;
    total_distracted_seconds: number;
    drowsiness_events: number;
    distraction_events: number;
    yawn_count: number;
    yawns_per_minute: number;
  };
}

const SafetyMonitor: React.FC = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [safetyState, setSafetyState] = useState<SafetyState | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<
    "disconnected" | "connecting" | "connected"
  >("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [activeTrip, setActiveTrip] = useState<{
    id: string;
    is_active: boolean;
  } | null>(null);
  const [autoStarted, setAutoStarted] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsRef = useRef(Date.now());
  const lastAlertRef = useRef(0);

  const vehicleId = "68f2ce4a-28df-4f11-bf5b-c961d1f7d064";

  const fetchActiveTrip = useCallback(async () => {
    try {
      const res = await fetch(
        `http://localhost:8000/api/telemetry/trips/active/${vehicleId}`
      );
      if (res.ok) {
        const trip = await res.json();
        if (trip?.is_active) {
          setActiveTrip(trip);
          return trip;
        }
      }
      setActiveTrip(null);
      return null;
    } catch {
      return null;
    }
  }, []);

  const playAlert = useCallback(() => {
    if (!soundEnabled) return;
    try {
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.5, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
      osc.start();
      osc.stop(ctx.currentTime + 0.3);
    } catch {}
  }, [soundEnabled]);

  const drawLandmarks = useCallback(
    (landmarks?: { left_eye: number[][]; right_eye: number[][] }) => {
      const canvas = overlayRef.current,
        video = videoRef.current;
      if (!canvas || !video) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!landmarks) return;
      const sx = canvas.width / 640,
        sy = canvas.height / 480;
      ctx.fillStyle = ORANGE;
      landmarks.left_eye?.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p[0] * sx, p[1] * sy, 3, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.fillStyle = "#fff";
      landmarks.right_eye?.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p[0] * sx, p[1] * sy, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    },
    []
  );

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraEnabled(true);
        setError(null);
      }
    } catch {
      setError("Camera access denied");
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraEnabled(false);
  }, []);

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current,
      canvas = canvasRef.current;
    if (!video || !canvas || video.readyState !== 4) return null;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.8);
  }, []);

  const startMonitoring = useCallback(() => {
    if (!cameraEnabled) {
      setError("Enable camera first");
      return;
    }
    setConnectionStatus("connecting");
    const ws = new WebSocket(
      `ws://localhost:8000/api/safety/stream/${vehicleId}`
    );
    wsRef.current = ws;
    ws.onopen = () => {
      console.log("[SAFETY-FE] WebSocket connected!");
      setConnectionStatus("connected");
      setError(null);
      setIsMonitoring(true);
      frameIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          const frame = captureFrame();
          if (frame) {
            console.log("[SAFETY-FE] Sending frame...");
            ws.send(JSON.stringify({ type: "frame", data: frame }));
          }
        }
      }, 100);
    };
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        console.log(
          "[SAFETY-FE] Received message:",
          msg.type,
          msg.data?.drowsiness?.face_detected
        );
        if (msg.type === "error") {
          // Handle CV not available error
          setError(msg.message || "CV module not available");
          setConnectionStatus("disconnected");
          setIsMonitoring(false);
          return;
        }
        if (msg.type === "detection") {
          setSafetyState(msg.data);
          drawLandmarks(msg.data?.drowsiness?.landmarks);
          const level = msg.data?.drowsiness?.alert_level;
          const now = Date.now();
          if (
            (level === "alert" || level === "critical") &&
            now - lastAlertRef.current > 500
          ) {
            playAlert();
            lastAlertRef.current = now;
          }
          frameCountRef.current++;
          if (now - lastFpsRef.current >= 1000) {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
            lastFpsRef.current = now;
          }
        }
      } catch {}
    };
    ws.onerror = () => setError("Connection error - is the backend running?");
    ws.onclose = () => {
      setConnectionStatus("disconnected");
      setIsMonitoring(false);
    };
  }, [cameraEnabled, captureFrame, drawLandmarks, playAlert]);

  const stopMonitoring = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "end" }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsMonitoring(false);
    setConnectionStatus("disconnected");
  }, []);

  useEffect(() => {
    const check = async () => {
      const trip = await fetchActiveTrip();
      if (trip?.is_active && cameraEnabled && !isMonitoring && !autoStarted) {
        startMonitoring();
        setAutoStarted(true);
      }
      if (!trip && isMonitoring && autoStarted) {
        stopMonitoring();
        setAutoStarted(false);
      }
    };
    check();
    const i = setInterval(check, 3000);
    return () => clearInterval(i);
  }, [
    fetchActiveTrip,
    cameraEnabled,
    isMonitoring,
    autoStarted,
    startMonitoring,
    stopMonitoring,
  ]);

  useEffect(
    () => () => {
      stopMonitoring();
      stopCamera();
    },
    [stopMonitoring, stopCamera]
  );

  const alertColor = (l: string) =>
    l === "critical"
      ? "#ef4444"
      : l === "alert"
      ? ORANGE
      : l === "warning"
      ? "#eab308"
      : "#22c55e";
  const scoreColor = (s: number) =>
    s >= 80 ? "#22c55e" : s >= 60 ? "#eab308" : s >= 40 ? ORANGE : "#ef4444";
  const d = safetyState?.drowsiness;
  const stats = safetyState?.session_stats;

  return (
    <div style={{ backgroundColor: DARK_BG, minHeight: "100vh", padding: 16 }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Shield size={24} style={{ color: ORANGE }} />
          <span style={{ color: DARK_TEXT, fontSize: 20, fontWeight: "bold" }}>
            Safety Monitor
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {activeTrip && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 10px",
                borderRadius: 20,
                backgroundColor: DARK_CARD,
                border: `1px solid ${DARK_BORDER}`,
              }}
            >
              <Car size={12} style={{ color: ORANGE }} />
              <span style={{ color: DARK_TEXT_MUTED, fontSize: 11 }}>
                Trip Active
              </span>
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  backgroundColor: "#22c55e",
                  animation: "pulse 2s infinite",
                }}
              />
            </div>
          )}
          <span style={{ color: DARK_TEXT_MUTED, fontSize: 11 }}>
            {fps} FPS
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor:
                  connectionStatus === "connected"
                    ? "#22c55e"
                    : connectionStatus === "connecting"
                    ? "#eab308"
                    : "#6b7280",
              }}
            />
            <span
              style={{
                color: DARK_TEXT_MUTED,
                fontSize: 11,
                textTransform: "capitalize",
              }}
            >
              {connectionStatus}
            </span>
          </div>
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            style={{
              padding: 6,
              borderRadius: 6,
              backgroundColor: soundEnabled ? ORANGE : DARK_CARD,
              color: soundEnabled ? "#000" : DARK_TEXT,
              border: "none",
              cursor: "pointer",
            }}
          >
            {soundEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>
        </div>
      </div>

      {error && (
        <div
          style={{
            marginBottom: 12,
            padding: 10,
            borderRadius: 8,
            backgroundColor: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.3)",
            color: DARK_TEXT,
            fontSize: 13,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <AlertTriangle size={14} style={{ color: "#ef4444" }} />
          {error}
        </div>
      )}

      {/* Main Layout - Side by Side */}
      <div style={{ display: "flex", gap: 12, height: "calc(100vh - 100px)" }}>
        {/* Left: Video Feed - Takes 70% */}
        <div
          style={{
            flex: "0 0 70%",
            backgroundColor: DARK_CARD,
            border: `1px solid ${DARK_BORDER}`,
            borderRadius: 12,
            padding: 12,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 8,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                color: DARK_TEXT,
                fontSize: 14,
                fontWeight: 500,
              }}
            >
              <Camera size={16} style={{ color: ORANGE }} />
              Camera
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={cameraEnabled ? stopCamera : startCamera}
                style={{
                  padding: "4px 12px",
                  borderRadius: 6,
                  fontSize: 12,
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  border: "none",
                  cursor: "pointer",
                  backgroundColor: cameraEnabled
                    ? "rgba(239,68,68,0.2)"
                    : ORANGE,
                  color: cameraEnabled ? "#ef4444" : "#000",
                }}
              >
                {cameraEnabled ? (
                  <>
                    <CameraOff size={14} />
                    Stop
                  </>
                ) : (
                  <>
                    <Camera size={14} />
                    Start
                  </>
                )}
              </button>
              <button
                onClick={isMonitoring ? stopMonitoring : startMonitoring}
                disabled={!cameraEnabled}
                style={{
                  padding: "4px 12px",
                  borderRadius: 6,
                  fontSize: 12,
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  border: "none",
                  cursor: cameraEnabled ? "pointer" : "not-allowed",
                  opacity: cameraEnabled ? 1 : 0.5,
                  backgroundColor: isMonitoring
                    ? "rgba(239,68,68,0.2)"
                    : "rgba(34,197,94,0.2)",
                  color: isMonitoring ? "#ef4444" : "#22c55e",
                }}
              >
                {isMonitoring ? (
                  <>
                    <Square size={14} />
                    Stop
                  </>
                ) : (
                  <>
                    <Play size={14} />
                    Monitor
                  </>
                )}
              </button>
            </div>
          </div>
          <div
            style={{
              flex: 1,
              position: "relative",
              backgroundColor: "#000",
              borderRadius: 8,
              overflow: "hidden",
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
            <canvas
              ref={overlayRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
              }}
            />
            <canvas ref={canvasRef} style={{ display: "none" }} />
            {d?.alert_level && d.alert_level !== "none" && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  border: `4px solid ${alertColor(d.alert_level)}`,
                  pointerEvents: "none",
                  animation:
                    d.alert_level === "critical" ? "pulse 1s infinite" : "none",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    top: 12,
                    left: "50%",
                    transform: "translateX(-50%)",
                    padding: "6px 16px",
                    borderRadius: 8,
                    backgroundColor: alertColor(d.alert_level),
                    color: "#000",
                    fontWeight: "bold",
                    fontSize: 14,
                  }}
                >
                  ⚠️ DROWSINESS {d.alert_level.toUpperCase()}
                </div>
              </div>
            )}
            {!cameraEnabled && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: DARK_CARD,
                }}
              >
                <div style={{ textAlign: "center" }}>
                  <CameraOff
                    size={48}
                    style={{
                      color: DARK_TEXT_MUTED,
                      opacity: 0.5,
                      marginBottom: 8,
                    }}
                  />
                  <p style={{ color: DARK_TEXT_MUTED, fontSize: 14 }}>
                    Click "Start" to enable camera
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: Stats Panel - Takes 30% */}
        <div
          style={{
            flex: "0 0 calc(30% - 12px)",
            display: "flex",
            flexDirection: "column",
            gap: 8,
            overflowY: "auto",
          }}
        >
          {/* Safety Score */}
          <div
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
              borderRadius: 12,
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: DARK_TEXT_MUTED,
                marginBottom: 4,
              }}
            >
              Safety Score
            </div>
            <div
              style={{
                fontSize: 40,
                fontWeight: "bold",
                color: scoreColor(safetyState?.safety_score ?? 100),
              }}
            >
              {Math.round(safetyState?.safety_score ?? 100)}
            </div>
          </div>

          {/* Eye Metrics */}
          <div
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
              borderRadius: 12,
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: DARK_TEXT_MUTED,
                marginBottom: 8,
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <Eye size={12} style={{ color: ORANGE }} />
              Eye Metrics
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                fontSize: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>State</span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  {d?.eyes_closed ? (
                    <>
                      <EyeOff size={14} style={{ color: "#ef4444" }} />
                      <span style={{ color: "#ef4444" }}>Closed</span>
                    </>
                  ) : (
                    <>
                      <Eye size={14} style={{ color: "#22c55e" }} />
                      <span style={{ color: "#22c55e" }}>Open</span>
                    </>
                  )}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>EAR</span>
                <span
                  style={{
                    color:
                      (d?.ear_average ?? 0.3) < 0.22 ? "#ef4444" : "#22c55e",
                  }}
                >
                  {(d?.ear_average ?? 0).toFixed(3)}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Blinks</span>
                <span style={{ color: DARK_TEXT }}>{d?.blink_count ?? 0}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Blinks/min</span>
                <span style={{ color: DARK_TEXT }}>
                  {(d?.blinks_per_minute ?? 0).toFixed(1)}
                </span>
              </div>
              {(d?.closed_duration_ms ?? 0) > 0 && (
                <div
                  style={{ display: "flex", justifyContent: "space-between" }}
                >
                  <span style={{ color: DARK_TEXT_MUTED }}>Closed</span>
                  <span style={{ color: "#ef4444" }}>
                    {((d?.closed_duration_ms ?? 0) / 1000).toFixed(1)}s
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Yawn Detection - NEW */}
          <div
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${d?.yawn?.is_yawning ? ORANGE : DARK_BORDER}`,
              borderRadius: 12,
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: DARK_TEXT_MUTED,
                marginBottom: 8,
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <Frown
                size={12}
                style={{
                  color: d?.yawn?.is_yawning ? ORANGE : DARK_TEXT_MUTED,
                }}
              />
              Yawn Detection
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                fontSize: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Status</span>
                <span
                  style={{
                    color: d?.yawn?.is_yawning ? ORANGE : "#22c55e",
                    fontWeight: d?.yawn?.is_yawning ? "bold" : "normal",
                  }}
                >
                  {d?.yawn?.is_yawning ? "🥱 YAWNING" : "Normal"}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>MAR</span>
                <span
                  style={{
                    color: (d?.yawn?.mar ?? 0) > 0.6 ? ORANGE : DARK_TEXT,
                  }}
                >
                  {(d?.yawn?.mar ?? 0).toFixed(3)}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Yawn Count</span>
                <span style={{ color: DARK_TEXT }}>
                  {d?.yawn?.yawn_count ?? 0}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Yawns/min</span>
                <span
                  style={{
                    color:
                      (d?.yawn?.yawns_per_minute ?? 0) > 0.4
                        ? ORANGE
                        : DARK_TEXT,
                  }}
                >
                  {(d?.yawn?.yawns_per_minute ?? 0).toFixed(2)}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Fatigue</span>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: 4,
                    fontSize: 10,
                    backgroundColor:
                      d?.yawn?.fatigue_level === "high"
                        ? "rgba(239,68,68,0.2)"
                        : d?.yawn?.fatigue_level === "moderate"
                        ? "rgba(234,179,8,0.2)"
                        : "rgba(34,197,94,0.2)",
                    color:
                      d?.yawn?.fatigue_level === "high"
                        ? "#ef4444"
                        : d?.yawn?.fatigue_level === "moderate"
                        ? "#eab308"
                        : "#22c55e",
                  }}
                >
                  {(d?.yawn?.fatigue_level ?? "low").toUpperCase()}
                </span>
              </div>
            </div>
          </div>

          {/* Distraction Detection - NEW */}
          <div
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${
                d?.distraction?.is_distracted ? "#ef4444" : DARK_BORDER
              }`,
              borderRadius: 12,
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: DARK_TEXT_MUTED,
                marginBottom: 8,
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <Focus
                size={12}
                style={{
                  color: d?.distraction?.is_distracted
                    ? "#ef4444"
                    : DARK_TEXT_MUTED,
                }}
              />
              Distraction Detection
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                fontSize: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Looking at Road</span>
                <span
                  style={{
                    color: d?.distraction?.looking_at_road
                      ? "#22c55e"
                      : "#ef4444",
                    fontWeight: !d?.distraction?.looking_at_road
                      ? "bold"
                      : "normal",
                  }}
                >
                  {d?.distraction?.looking_at_road ? "✓ Yes" : "✗ No"}
                </span>
              </div>
              {d?.distraction?.distraction_type &&
                d.distraction.distraction_type !== "none" && (
                  <div
                    style={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <span style={{ color: DARK_TEXT_MUTED }}>Type</span>
                    <span style={{ color: "#ef4444" }}>
                      {d.distraction.distraction_type
                        .replace("_", " ")
                        .toUpperCase()}
                    </span>
                  </div>
                )}
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Attention</span>
                <span
                  style={{
                    color:
                      (d?.distraction?.attention_score ?? 100) >= 80
                        ? "#22c55e"
                        : (d?.distraction?.attention_score ?? 100) >= 50
                        ? "#eab308"
                        : "#ef4444",
                  }}
                >
                  {(d?.distraction?.attention_score ?? 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Head Pose</span>
                <span style={{ color: DARK_TEXT, fontSize: 10 }}>
                  P:{(d?.distraction?.pitch ?? 0).toFixed(0)}° Y:
                  {(d?.distraction?.yaw ?? 0).toFixed(0)}° R:
                  {(d?.distraction?.roll ?? 0).toFixed(0)}°
                </span>
              </div>
              {(d?.distraction?.distraction_duration_ms ?? 0) > 0 && (
                <div
                  style={{ display: "flex", justifyContent: "space-between" }}
                >
                  <span style={{ color: DARK_TEXT_MUTED }}>Duration</span>
                  <span style={{ color: "#ef4444" }}>
                    {(
                      (d?.distraction?.distraction_duration_ms ?? 0) / 1000
                    ).toFixed(1)}
                    s
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Alert Status */}
          <div
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
              borderRadius: 12,
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: DARK_TEXT_MUTED,
                marginBottom: 8,
              }}
            >
              Alert Status
            </div>
            <div
              style={{
                padding: 8,
                borderRadius: 6,
                textAlign: "center",
                fontWeight: "bold",
                fontSize: 14,
                backgroundColor: alertColor(d?.alert_level ?? "none"),
                color: "#000",
              }}
            >
              {(d?.alert_level ?? "none").toUpperCase()}
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 12,
                marginTop: 8,
              }}
            >
              <span style={{ color: DARK_TEXT_MUTED }}>Face</span>
              <span style={{ color: d?.face_detected ? "#22c55e" : "#ef4444" }}>
                {d?.face_detected ? "Yes" : "No"}
              </span>
            </div>
          </div>

          {/* Session Stats */}
          <div
            style={{
              backgroundColor: DARK_CARD,
              border: `1px solid ${DARK_BORDER}`,
              borderRadius: 12,
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: DARK_TEXT_MUTED,
                marginBottom: 8,
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <Activity size={12} style={{ color: ORANGE }} />
              Session Stats
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 4,
                fontSize: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Duration</span>
                <span style={{ color: DARK_TEXT }}>
                  {Math.floor((stats?.duration_seconds ?? 0) / 60)}:
                  {String(
                    Math.floor((stats?.duration_seconds ?? 0) % 60)
                  ).padStart(2, "0")}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Drowsy Time</span>
                <span style={{ color: ORANGE }}>
                  {(stats?.total_drowsy_seconds ?? 0).toFixed(1)}s
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Distracted Time</span>
                <span style={{ color: "#ef4444" }}>
                  {(stats?.total_distracted_seconds ?? 0).toFixed(1)}s
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Drowsy Events</span>
                <span style={{ color: ORANGE }}>
                  {stats?.drowsiness_events ?? 0}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>
                  Distraction Events
                </span>
                <span style={{ color: "#ef4444" }}>
                  {stats?.distraction_events ?? 0}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: DARK_TEXT_MUTED }}>Total Yawns</span>
                <span style={{ color: DARK_TEXT }}>
                  {stats?.yawn_count ?? 0}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SafetyMonitor;
