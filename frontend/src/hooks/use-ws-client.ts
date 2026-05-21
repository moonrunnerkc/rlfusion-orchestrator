// Author: Bradley R. Kinnard
// WS client hook with exponential-backoff reconnect (1s..30s).

import { useEffect, useRef, useState } from 'react';

export type WsStatus = 'connecting' | 'open' | 'closed' | 'error';

export interface WsClient {
  status: WsStatus;
  send: (data: unknown) => void;
}

interface Options {
  url: string;
  onMessage: (data: unknown) => void;
}

const BACKOFF_BASE_MS = 1000;
const BACKOFF_MAX_MS = 30_000;

export function useWsClient({ url, onMessage }: Options): WsClient {
  const [status, setStatus] = useState<WsStatus>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const handlerRef = useRef(onMessage);

  useEffect(() => {
    handlerRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (!alive) return;
      setStatus('connecting');
      const socket = new WebSocket(url);
      wsRef.current = socket;

      socket.onopen = () => {
        retriesRef.current = 0;
        setStatus('open');
      };
      socket.onmessage = (e) => {
        try {
          handlerRef.current(JSON.parse(e.data));
        } catch {
          // ignore malformed frames; backend wraps everything in JSON.
        }
      };
      socket.onerror = () => {
        setStatus('error');
      };
      socket.onclose = () => {
        if (!alive) return;
        setStatus('closed');
        const delay = Math.min(
          BACKOFF_MAX_MS,
          BACKOFF_BASE_MS * 2 ** retriesRef.current,
        );
        retriesRef.current += 1;
        timer = setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
      wsRef.current?.close();
    };
  }, [url]);

  return {
    status,
    send: (data) => {
      const ws = wsRef.current;
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
      }
    },
  };
}
