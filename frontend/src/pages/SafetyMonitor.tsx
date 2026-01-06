import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Eye,
  EyeOff,
  AlertTriangle,
  AlertCircle,
  Camera,
  CameraOff,
  Activity,
  Shield,
  ShieldAlert,
  Volume2,
  VolumeX,
  Play,
  Square,
} from "lucide-react";

// Types
interface DrowsinessState {
  ear_left: number;
  ear_right: number;
  ear_average: number;
  eyes_closed: boolean;
  closed_duration_ms: number;
  blink_count: number;
  blinks_per_minute: number;
  alert_level: "none" | "warning" | "alert" | "critical";
  is_drowsy: boolean;
  head_pitch: number | null;
  head_yaw: number | null;
  head_roll: number | null;
  face_detected: boolean;
  confidence: number;
  timestamp: number;
}

interface SafetyState {
  drowsiness: DrowsinessState;
  distraction?: {
    head_pose: { pitch: number; yaw: number; roll: number };
    is_distracted: boolean;
    distraction_type: string;
    distraction_duration_ms: number;
    looking_at_road: boolean;
    attention_score: number;
  };
  is_safe: boolean;
  safety_score: number;
  active_alerts: string[];
  session_stats: {
    duration_seconds: number;
    total_drowsy_seconds: number;
    total_distracted_seconds: number;
    drowsiness_events: number;
    distraction_events: number;
  };
}

interface SafetyAlert {
  type: string;
  severity: string;
  timestamp: string;
  duration_seconds: number;
  details: Record<string, unknown>;
}

const SafetyMonitor: React.FC = () => {
  // State
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [safetyState, setSafetyState] = useState<SafetyState | null>(null);
  const [alerts, setAlerts] = useState<SafetyAlert[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<
    "disconnected" | "connecting" | "connected"
  >("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  // Vehicle ID (you'd get this from your app context)
  const vehicleId = "68f2ce4a-28df-4f11-bf5b-c961d1f7d064";

  // Play alert sound using Web Audio API
  const playAlertSound = useCallback(() => {
    if (soundEnabled) {
      try {
        const audioContext = new (window.AudioContext ||
          (window as any).webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 880;
        oscillator.type = "sine";
        gainNode.gain.value = 0.3;

        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.2);
      } catch (e) {
        console.log("Audio not available");
      }
    }
  }, [soundEnabled]);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      streamRef.current = stream;
      setCameraEnabled(true);
      setError(null);
    } catch (err) {
      setError("Failed to access camera. Please allow camera permissions.");
      console.error("Camera error:", err);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraEnabled(false);
  }, []);

  // Capture frame as base64
  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx || video.videoWidth === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    return canvas.toDataURL("image/jpeg", 0.8);
  }, []);

  // Connect WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionStatus("connecting");
    const ws = new WebSocket(
      `ws://localhost:8000/api/safety/stream/${vehicleId}`
    );

    ws.onopen = () => {
      setConnectionStatus("connected");
      setError(null);
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      console.log("WS message:", message); // ADD THIS
      if (message.type === "detection") {
        setSafetyState(message.data);

        // Draw landmarks on overlay
        drawLandmarks(message.data?.drowsiness?.landmarks);

        // Play alert sound while drowsy (throttled to every 500ms)
        const alertLevel = message.data?.drowsiness?.alert_level;
        const now = Date.now();
        if (
          (alertLevel === "alert" || alertLevel === "critical") &&
          now - lastAlertSoundRef.current > 500
        ) {
          playAlertSound();
          lastAlertSoundRef.current = now;
        }

        frameCountRef.current++;
        if (now - lastFpsUpdateRef.current >= 1000) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastFpsUpdateRef.current = now;
        }

        const state = message.data as SafetyState;
        if (
          state.drowsiness.alert_level === "alert" ||
          state.drowsiness.alert_level === "critical"
        ) {
          playAlertSound();
        }
      } else if (message.type === "alert") {
        setAlerts((prev) => [message.data, ...prev.slice(0, 49)]);
        playAlertSound();
      }
    };

    ws.onerror = () =>
      setError("Connection error. Make sure the backend is running.");
    ws.onclose = () => setConnectionStatus("disconnected");

    wsRef.current = ws;
  }, [vehicleId, playAlertSound]);

  // Draw landmarks on overlay canvas
  const drawLandmarks = useCallback(
    (
      landmarks: { left_eye: number[][]; right_eye: number[][] } | undefined
    ) => {
      const video = videoRef.current;
      const overlay = overlayCanvasRef.current;
      if (!overlay || !video) return;

      const ctx = overlay.getContext("2d");
      if (!ctx) return;

      // Match canvas size to video display size
      const rect = video.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;

      // Clear previous drawings
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      if (!landmarks) return;

      // Scale factors (video resolution to display size)
      const scaleX = rect.width / video.videoWidth;
      const scaleY = rect.height / video.videoHeight;

      // Draw function for eye points
      const drawEye = (points: number[][], color: string) => {
        if (!points || points.length === 0) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.fillStyle = color;

        // Draw points
        points.forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(x * scaleX, y * scaleY, 3, 0, 2 * Math.PI);
          ctx.fill();
        });

        // Draw connecting lines
        ctx.beginPath();
        ctx.moveTo(points[0][0] * scaleX, points[0][1] * scaleY);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i][0] * scaleX, points[i][1] * scaleY);
        }
        ctx.closePath();
        ctx.stroke();
      };

      // Draw left eye (green) and right eye (blue)
      drawEye(landmarks.left_eye, "#00ff00");
      drawEye(landmarks.right_eye, "#00aaff");
    },
    []
  );

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "end" }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus("disconnected");
  }, []);

  // Send frame to server
  const sendFrame = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const frame = captureFrame();
    if (frame) {
      wsRef.current.send(JSON.stringify({ type: "frame", data: frame }));
    }
  }, [captureFrame]);

  // Start monitoring
  const startMonitoring = useCallback(async () => {
    if (!cameraEnabled) await startCamera();
    connectWebSocket();
    frameIntervalRef.current = setInterval(sendFrame, 66); // ~15 FPS
    setIsMonitoring(true);
  }, [cameraEnabled, startCamera, connectWebSocket, sendFrame]);

  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    disconnectWebSocket();
    setIsMonitoring(false);
  }, [disconnectWebSocket]);

  // Cleanup
  useEffect(() => {
    return () => {
      stopMonitoring();
      stopCamera();
    };
  }, [stopMonitoring, stopCamera]);

  // Helpers
  const getAlertColor = (level: string) => {
    switch (level) {
      case "critical":
        return "bg-red-500";
      case "alert":
        return "bg-orange-500";
      case "warning":
        return "bg-yellow-500";
      default:
        return "bg-green-500";
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-500";
    if (score >= 60) return "text-yellow-500";
    if (score >= 40) return "text-orange-500";
    return "text-red-500";
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3">
              <Shield className="w-8 h-8 text-blue-500" />
              Driver Safety Monitor
            </h1>
            <p className="text-gray-400 mt-1">
              Real-time drowsiness & distraction detection
            </p>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`p-2 rounded-lg ${
                soundEnabled ? "bg-blue-600" : "bg-gray-700"
              }`}
            >
              {soundEnabled ? (
                <Volume2 className="w-5 h-5" />
              ) : (
                <VolumeX className="w-5 h-5" />
              )}
            </button>

            <div
              className={`px-3 py-1 rounded-full text-sm flex items-center gap-2 ${
                connectionStatus === "connected"
                  ? "bg-green-600"
                  : connectionStatus === "connecting"
                  ? "bg-yellow-600"
                  : "bg-gray-700"
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  connectionStatus === "connected"
                    ? "bg-green-300 animate-pulse"
                    : connectionStatus === "connecting"
                    ? "bg-yellow-300 animate-pulse"
                    : "bg-gray-500"
                }`}
              />
              {connectionStatus}
            </div>

            {isMonitoring && (
              <div className="px-3 py-1 bg-gray-800 rounded-full text-sm">
                {fps} FPS
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 mb-6 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span>{error}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera Feed */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Camera className="w-5 h-5" />
                  Camera Feed
                </h2>
                <div className="flex gap-2">
                  {!cameraEnabled ? (
                    <button
                      onClick={startCamera}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2"
                    >
                      <Camera className="w-4 h-4" /> Enable Camera
                    </button>
                  ) : !isMonitoring ? (
                    <button
                      onClick={startMonitoring}
                      className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2"
                    >
                      <Play className="w-4 h-4" /> Start Monitoring
                    </button>
                  ) : (
                    <button
                      onClick={stopMonitoring}
                      className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center gap-2"
                    >
                      <Square className="w-4 h-4" /> Stop
                    </button>
                  )}
                </div>
              </div>

              <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  playsInline
                  muted
                />
                <canvas ref={canvasRef} className="hidden" />
                <canvas
                  ref={overlayCanvasRef}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                />

                {safetyState && (
                  <>
                    {safetyState.drowsiness.alert_level !== "none" && (
                      <div
                        className={`absolute inset-0 pointer-events-none ${
                          safetyState.drowsiness.alert_level === "critical"
                            ? "bg-red-500/30 animate-pulse"
                            : safetyState.drowsiness.alert_level === "alert"
                            ? "bg-orange-500/20"
                            : "bg-yellow-500/10"
                        }`}
                      />
                    )}

                    {!safetyState.drowsiness.face_detected && (
                      <div className="absolute top-4 right-4 bg-black/70 px-3 py-2 rounded-lg flex items-center gap-2">
                        <CameraOff className="w-4 h-4 text-yellow-400" />
                        <span className="text-yellow-400 text-sm">
                          Face not detected
                        </span>
                      </div>
                    )}

                    <div className="absolute top-4 left-4">
                      <div
                        className={`px-3 py-1 rounded-full text-sm flex items-center gap-2 ${
                          safetyState.drowsiness.eyes_closed
                            ? "bg-red-600"
                            : "bg-green-600"
                        }`}
                      >
                        {safetyState.drowsiness.eyes_closed ? (
                          <EyeOff className="w-4 h-4" />
                        ) : (
                          <Eye className="w-4 h-4" />
                        )}
                        {safetyState.drowsiness.eyes_closed ? "Closed" : "Open"}
                      </div>
                    </div>

                    {safetyState.drowsiness.is_drowsy && (
                      <div className="absolute bottom-0 left-0 right-0 bg-red-600 py-3 px-4 flex items-center justify-center gap-3 animate-pulse">
                        <AlertTriangle className="w-6 h-6" />
                        <span className="font-bold text-lg">
                          DROWSINESS DETECTED!
                        </span>
                        <AlertTriangle className="w-6 h-6" />
                      </div>
                    )}
                  </>
                )}

                {!cameraEnabled && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                    <div className="text-center">
                      <CameraOff className="w-16 h-16 mx-auto text-gray-600 mb-4" />
                      <p className="text-gray-400 mb-4">Camera is disabled</p>
                      <button
                        onClick={startCamera}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg"
                      >
                        Enable Camera
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Stats Panel */}
          <div className="space-y-4">
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                {safetyState?.is_safe ? (
                  <Shield className="w-5 h-5 text-green-500" />
                ) : (
                  <ShieldAlert className="w-5 h-5 text-red-500" />
                )}
                Safety Score
              </h3>
              <div className="text-center">
                <div
                  className={`text-6xl font-bold ${getScoreColor(
                    safetyState?.safety_score ?? 100
                  )}`}
                >
                  {safetyState?.safety_score?.toFixed(0) ?? "--"}
                </div>
                <div className="text-gray-400 mt-2">/ 100</div>
              </div>
              {safetyState && safetyState.drowsiness.alert_level !== "none" && (
                <div
                  className={`mt-4 p-3 rounded-lg text-center ${
                    safetyState.drowsiness.alert_level === "critical"
                      ? "bg-red-600"
                      : safetyState.drowsiness.alert_level === "alert"
                      ? "bg-orange-600"
                      : "bg-yellow-600"
                  }`}
                >
                  <span className="font-semibold uppercase">
                    {safetyState.drowsiness.alert_level} Level
                  </span>
                </div>
              )}
            </div>

            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5" /> Eye Metrics
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Eye Aspect Ratio</span>
                    <span>
                      {safetyState?.drowsiness.ear_average.toFixed(3) ?? "--"}
                    </span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        (safetyState?.drowsiness.ear_average ?? 0.3) < 0.22
                          ? "bg-red-500"
                          : "bg-green-500"
                      }`}
                      style={{
                        width: `${Math.min(
                          100,
                          ((safetyState?.drowsiness.ear_average ?? 0.3) / 0.4) *
                            100
                        )}%`,
                      }}
                    />
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Blinks/min</span>
                  <span
                    className={`text-xl font-semibold ${
                      (safetyState?.drowsiness.blinks_per_minute ?? 0) > 25
                        ? "text-yellow-500"
                        : "text-white"
                    }`}
                  >
                    {safetyState?.drowsiness.blinks_per_minute.toFixed(1) ??
                      "--"}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Eyes Closed</span>
                  <span
                    className={`text-xl font-semibold ${
                      (safetyState?.drowsiness.closed_duration_ms ?? 0) > 500
                        ? "text-red-500"
                        : "text-white"
                    }`}
                  >
                    {(
                      (safetyState?.drowsiness.closed_duration_ms ?? 0) / 1000
                    ).toFixed(1)}
                    s
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5" /> Session Stats
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Duration</span>
                  <span>
                    {formatDuration(
                      safetyState?.session_stats.duration_seconds ?? 0
                    )}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Drowsy Time</span>
                  <span className="text-orange-400">
                    {(
                      safetyState?.session_stats.total_drowsy_seconds ?? 0
                    ).toFixed(1)}
                    s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Drowsiness Events</span>
                  <span>
                    {safetyState?.session_stats.drowsiness_events ?? 0}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Alerts */}
        <div className="mt-6 bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" /> Recent Alerts
            {alerts.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-red-600 rounded-full text-xs">
                {alerts.length}
              </span>
            )}
          </h3>
          {alerts.length === 0 ? (
            <p className="text-gray-400 text-center py-8">
              No alerts yet. Stay safe!
            </p>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {alerts.map((alert, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-lg flex items-center gap-3 ${
                    alert.severity === "critical"
                      ? "bg-red-900/50 border border-red-500"
                      : alert.severity === "warning"
                      ? "bg-orange-900/50 border border-orange-500"
                      : "bg-gray-700"
                  }`}
                >
                  <div
                    className={`w-2 h-2 rounded-full ${getAlertColor(
                      alert.severity
                    )}`}
                  />
                  <div className="flex-1">
                    <div className="font-medium">
                      {alert.type?.replace(/_/g, " ") ?? "Unknown Alert"}
                    </div>
                    <div className="text-sm text-gray-400">
                      Duration: {alert.duration_seconds.toFixed(1)}s
                    </div>
                  </div>
                  <div className="text-sm text-gray-400">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SafetyMonitor;
