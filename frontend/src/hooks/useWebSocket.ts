import { useState, useEffect, useCallback, useRef } from 'react';
import { WebSocketMessage } from '../types';
import { 
  createAppError, 
  ErrorCategory, 
  ErrorSeverity, 
  AppError,
  logError,
  gracefulDegradation
} from '../utils/errorHandling';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: AppError) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
}

interface UseWebSocketReturn {
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => boolean;
  connect: () => void;
  disconnect: () => void;
  reconnectAttempts: number;
  error: AppError | null;
  isHealthy: boolean;
}

export const useWebSocket = ({
  url,
  onMessage,
  onError,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5,
  heartbeatInterval = 30000,
  connectionTimeout = 10000
}: UseWebSocketOptions): UseWebSocketReturn => {
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [error, setError] = useState<AppError | null>(null);
  const [isHealthy, setIsHealthy] = useState(true);
  
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef(true);
  const lastHeartbeatRef = useRef<number>(Date.now());

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }
  }, []);

  const handleError = useCallback((errorMessage: string, originalError?: Error, severity: ErrorSeverity = ErrorSeverity.MEDIUM) => {
    const appError = createAppError(
      errorMessage,
      ErrorCategory.WEBSOCKET,
      severity,
      {
        originalError,
        details: {
          url,
          reconnectAttempts,
          timestamp: new Date().toISOString()
        }
      }
    );
    
    setError(appError);
    setConnectionStatus('error');
    setIsHealthy(false);
    logError(appError);
    onError?.(appError);
    
    // Update service status
    gracefulDegradation.setServiceStatus('websocket', false);
  }, [url, reconnectAttempts, onError]);

  const startHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
    }
    
    heartbeatTimeoutRef.current = setTimeout(() => {
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        try {
          websocketRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
          lastHeartbeatRef.current = Date.now();
          startHeartbeat(); // Schedule next heartbeat
        } catch (error) {
          handleError('Failed to send heartbeat', error as Error, ErrorSeverity.LOW);
        }
      }
    }, heartbeatInterval);
  }, [heartbeatInterval, handleError]);

  const connect = useCallback(() => {
    // Don't connect if URL is empty or invalid
    if (!url || url.trim() === '') {
      setConnectionStatus('disconnected');
      return;
    }
    
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Clear any existing timeouts
    clearTimeouts();
    
    setConnectionStatus('connecting');
    setError(null);
    
    // Set connection timeout
    connectionTimeoutRef.current = setTimeout(() => {
      if (connectionStatus === 'connecting') {
        handleError('Connection timeout', undefined, ErrorSeverity.HIGH);
        if (websocketRef.current) {
          websocketRef.current.close();
        }
      }
    }, connectionTimeout);
    
    try {
      const ws = new WebSocket(url);
      websocketRef.current = ws;

      ws.onopen = () => {
        clearTimeouts();
        setConnectionStatus('connected');
        setReconnectAttempts(0);
        setError(null);
        setIsHealthy(true);
        shouldReconnectRef.current = true;
        
        // Update service status
        gracefulDegradation.setServiceStatus('websocket', true);
        
        // Start heartbeat
        startHeartbeat();
        
        console.log('WebSocket connected successfully');
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Handle pong responses
          if (message.type === 'pong') {
            lastHeartbeatRef.current = Date.now();
            return;
          }
          
          setLastMessage(message);
          onMessage?.(message);
          
          // Update health status
          setIsHealthy(true);
          
        } catch (error) {
          handleError('Failed to parse WebSocket message', error as Error, ErrorSeverity.LOW);
        }
      };

      ws.onclose = (event) => {
        clearTimeouts();
        setConnectionStatus('disconnected');
        websocketRef.current = null;
        
        // Update service status
        gracefulDegradation.setServiceStatus('websocket', false);
        
        const wasCleanClose = event.code === 1000;
        const reason = event.reason || 'Connection closed';
        
        if (!wasCleanClose && shouldReconnectRef.current) {
          console.warn(`WebSocket closed unexpectedly: ${reason} (code: ${event.code})`);
          
          // Attempt to reconnect if enabled and under max attempts
          if (reconnectAttempts < maxReconnectAttempts) {
            const delay = Math.min(reconnectInterval * Math.pow(2, reconnectAttempts), 30000);
            
            console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              setReconnectAttempts(prev => prev + 1);
              connect();
            }, delay);
          } else {
            handleError(
              `Failed to reconnect after ${maxReconnectAttempts} attempts`,
              undefined,
              ErrorSeverity.HIGH
            );
          }
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        handleError('WebSocket connection error', undefined, ErrorSeverity.HIGH);
      };

    } catch (error) {
      handleError('Failed to create WebSocket connection', error as Error, ErrorSeverity.CRITICAL);
    }
  }, [
    url, 
    connectionTimeout, 
    reconnectInterval, 
    maxReconnectAttempts, 
    reconnectAttempts, 
    connectionStatus,
    onMessage, 
    handleError, 
    clearTimeouts, 
    startHeartbeat
  ]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    clearTimeouts();

    if (websocketRef.current) {
      websocketRef.current.close(1000, 'User initiated disconnect');
      websocketRef.current = null;
    }
    
    setConnectionStatus('disconnected');
    setReconnectAttempts(0);
    setError(null);
    setIsHealthy(true);
    
    // Update service status
    gracefulDegradation.setServiceStatus('websocket', true);
  }, [clearTimeouts]);

  const sendMessage = useCallback((message: any): boolean => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      try {
        websocketRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        handleError('Failed to send WebSocket message', error as Error, ErrorSeverity.MEDIUM);
        return false;
      }
    } else {
      console.warn('WebSocket is not connected. Cannot send message.');
      return false;
    }
  }, [handleError]);

  // Monitor connection health
  useEffect(() => {
    if (connectionStatus === 'connected') {
      const healthCheckInterval = setInterval(() => {
        const timeSinceLastHeartbeat = Date.now() - lastHeartbeatRef.current;
        
        if (timeSinceLastHeartbeat > heartbeatInterval * 2) {
          console.warn('WebSocket appears unhealthy - no recent heartbeat');
          setIsHealthy(false);
          
          // Attempt to reconnect if connection seems dead
          if (timeSinceLastHeartbeat > heartbeatInterval * 3) {
            handleError('Connection appears dead - no heartbeat response', undefined, ErrorSeverity.HIGH);
            if (websocketRef.current) {
              websocketRef.current.close();
            }
          }
        }
      }, heartbeatInterval);
      
      return () => clearInterval(healthCheckInterval);
    }
  }, [connectionStatus, heartbeatInterval, handleError]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connectionStatus,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    reconnectAttempts,
    error,
    isHealthy
  };
};