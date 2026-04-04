import { useState, useEffect, useRef, useCallback } from 'react';

export interface Vehicle {
  id: string;
  x: number;
  y: number;
  phi: number;
  type: string;
}

export interface TelemetryData {
  v2: {
    vehicles: Vehicle[];
    tls: Record<string, string>;
    reward: number;
    step: number;
  };
  native: {
    vehicles: Vehicle[];
    tls: Record<string, string>;
    reward: number;
    step: number;
  };
}

export function useTrafficSocket(url: string) {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    socketRef.current = new WebSocket(url);

    socketRef.current.onopen = () => {
      console.log('Connected to Traffic Telemetry');
      setIsConnected(true);
    };

    socketRef.current.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.v2 && payload.native) {
          setData(payload);
        }
      } catch (e) {
        console.error('Error parsing telemetry', e);
      }
    };

    socketRef.current.onclose = () => {
      console.log('Disconnected from Traffic Telemetry');
      setIsConnected(false);
      setIsRunning(false);
    };

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [url]);

  const startSimulation = () => {
    setData(null);
    if (socketRef.current && isConnected) {
      socketRef.current.send(JSON.stringify({ command: 'start' }));
      setIsRunning(true);
    }
  };

  const stopSimulation = useCallback(() => {
    if (socketRef.current && isConnected) {
      socketRef.current.send(JSON.stringify({ command: 'stop' }));
      setIsRunning(false);
    }
  }, [isConnected]);

  return { data, isConnected, isRunning, startSimulation, stopSimulation };
}
