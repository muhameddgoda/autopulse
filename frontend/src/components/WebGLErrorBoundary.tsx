import React, { Component, ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error boundary specifically for WebGL/Three.js components.
 * Shows a fallback UI instead of crashing the whole app.
 */
export class WebGLErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('WebGL Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback or default
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="w-full h-full flex items-center justify-center bg-gray-800/50 rounded-xl">
          <div className="text-center p-8">
            <AlertTriangle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">
              3D View Unavailable
            </h3>
            <p className="text-gray-400 max-w-md">
              WebGL is not available in your browser. This could be due to:
            </p>
            <ul className="text-gray-400 text-sm mt-2 space-y-1">
              <li>• Hardware acceleration disabled</li>
              <li>• Running in a VM or remote desktop</li>
              <li>• Outdated graphics drivers</li>
              <li>• Browser restrictions</li>
            </ul>
            <p className="text-gray-500 text-sm mt-4">
              The dashboard will work without the 3D view.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Check if WebGL is available before rendering
 */
export function isWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch (e) {
    return false;
  }
}

/**
 * Check if WebGL2 is available
 */
export function isWebGL2Available(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && canvas.getContext('webgl2'));
  } catch (e) {
    return false;
  }
}

export default WebGLErrorBoundary;
