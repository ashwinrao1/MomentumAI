/**
 * API Configuration for MomentumML Frontend
 */

// Determine the API base URL based on environment
const getApiBaseUrl = (): string => {
  // Check if we're in development mode
  if (process.env.NODE_ENV === 'development') {
    // Use environment variable if set, otherwise default to localhost:8003
    return process.env.REACT_APP_API_URL || 'http://localhost:8003';
  }
  
  // In production, use relative URLs or environment variable
  return process.env.REACT_APP_API_URL || '';
};

// API Configuration
export const API_CONFIG = {
  BASE_URL: getApiBaseUrl(),
  ENDPOINTS: {
    GAMES: '/api/momentum/games',
    CURRENT_MOMENTUM: '/api/momentum/current',
    STATUS: '/api/momentum/status',
    VISUALIZATION: '/api/games/{gameId}/momentum/visualization',
    DEMO_VISUALIZATION: '/api/games/demo/momentum/visualization'
  },
  WEBSOCKET: {
    BASE_URL: getApiBaseUrl().replace('http', 'ws'),
    LIVE_STREAM: '/live/stream'
  }
};

// Helper functions
export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

export const getWebSocketUrl = (endpoint: string): string => {
  return `${API_CONFIG.WEBSOCKET.BASE_URL}${endpoint}`;
};

export const getVisualizationUrl = (gameId: string): string => {
  if (gameId === 'demo_game_001') {
    return getApiUrl(API_CONFIG.ENDPOINTS.DEMO_VISUALIZATION);
  }
  return getApiUrl(API_CONFIG.ENDPOINTS.VISUALIZATION.replace('{gameId}', gameId));
};