import { useEffect, useRef } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { MockTrainingStream } from '../lib/mockStream';
import { WebSocketMessage } from '../types';

const WS_URL = 'ws://localhost:8000/ws/metrics';

export function useTrainingStream() {
  const {
    isMockMode,
    handleWebSocketMessage,
    setConnectionStatus,
  } = useDashboardStore();

  const wsRef = useRef<WebSocket | null>(null);
  const mockStreamRef = useRef<MockTrainingStream | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    // 1. If in mock mode, use simulated live stream
    if (isMockMode) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnectionStatus('mock');

      if (!mockStreamRef.current) {
        mockStreamRef.current = new MockTrainingStream((msg: WebSocketMessage) => {
          handleWebSocketMessage(msg);
        });
        mockStreamRef.current.start();
      }

      return () => {
        if (mockStreamRef.current) {
          mockStreamRef.current.stop();
          mockStreamRef.current = null;
        }
      };
    }

    // 2. Otherwise connect to real WebSocket
    if (mockStreamRef.current) {
      mockStreamRef.current.stop();
      mockStreamRef.current = null;
    }

    let isUnmounted = false;

    function connect() {
      if (isUnmounted) return;

      try {
        setConnectionStatus('reconnecting');
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          if (isUnmounted) return;
          setConnectionStatus('connected');
        };

        ws.onmessage = (event) => {
          if (isUnmounted) return;
          try {
            const data = JSON.parse(event.data);
            // Translate server payload format if needed
            if (data.global_val_dice !== undefined) {
              handleWebSocketMessage({
                type: 'metrics',
                round: data.round,
                global_loss: data.global_val_loss ?? 0.35,
                dice: data.global_val_dice,
                baseline_dice: 0.845,
              });
            } else if (data.type) {
              handleWebSocketMessage(data as WebSocketMessage);
            }
          } catch (e) {
            console.error('Error parsing live WS metrics:', e);
          }
        };

        ws.onerror = () => {
          if (isUnmounted) return;
          setConnectionStatus('reconnecting');
        };

        ws.onclose = () => {
          if (isUnmounted) return;
          setConnectionStatus('reconnecting');
          reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
        };
      } catch (err) {
        setConnectionStatus('reconnecting');
        reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
      }
    }

    connect();

    return () => {
      isUnmounted = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [isMockMode, handleWebSocketMessage, setConnectionStatus]);
}
