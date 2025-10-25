// Core data interfaces for frontend TypeScript

export interface GameEvent {
  event_id: string;
  game_id: string;
  team_tricode: string;
  player_name?: string;
  event_type: string; // shot, rebound, turnover, foul
  clock: string;
  period: number;
  points_total: number;
  shot_result?: string; // Made, Missed
  timestamp: string;
}

export interface Possession {
  possession_id: string;
  game_id: string;
  team_tricode: string;
  start_time: string;
  end_time: string;
  points_scored: number;
  fg_attempts: number;
  fg_made: number;
  turnovers: number;
  rebounds: number;
  fouls: number;
}

export interface TeamMomentumIndex {
  game_id: string;
  team_tricode: string;
  timestamp: string;
  tmi_value: number;
  feature_contributions: Record<string, number>;
  rolling_window_size: number;
  prediction_probability: number;
  confidence_score: number;
}

export interface GameSelection {
  game_id: string;
  home_team: string;
  away_team: string;
  game_date: string;
  status: string;
}

export interface MomentumData {
  home_team: TeamMomentumIndex;
  away_team: TeamMomentumIndex;
  game_time: string;
  quarter: number;
  home_score: number;
  away_score: number;
}

export interface WebSocketMessage {
  type: 'momentum_update' | 'game_state' | 'error' | 'pong';
  data: any;
  timestamp: string;
}

export interface ConnectionStatus {
  status: 'connecting' | 'connected' | 'disconnected';
  lastConnected?: string;
  reconnectAttempts?: number;
}

export interface UserConfiguration {
  rollingWindowSize: number;
  refreshRate: number; // in seconds
  predictionConfidenceThreshold: number; // 0-1
  chartUpdateInterval: number; // in milliseconds
  maxHistoryPoints: number;
}

export interface ConfigurationControlsProps {
  configuration: UserConfiguration;
  onConfigurationChange: (config: UserConfiguration) => void;
  isOpen: boolean;
  onToggle: () => void;
}