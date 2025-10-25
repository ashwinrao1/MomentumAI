import { UserConfiguration } from '../types';

const STORAGE_KEY = 'momentumml_user_config';

// Default configuration values
export const DEFAULT_CONFIG: UserConfiguration = {
  rollingWindowSize: 5, // possessions
  refreshRate: 25, // seconds
  predictionConfidenceThreshold: 0.6, // 60%
  chartUpdateInterval: 1000, // 1 second
  maxHistoryPoints: 50
};

// Load user configuration from localStorage
export const loadUserConfiguration = (): UserConfiguration => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Merge with defaults to handle missing properties
      return { ...DEFAULT_CONFIG, ...parsed };
    }
  } catch (error) {
    console.warn('Failed to load user configuration from localStorage:', error);
  }
  return DEFAULT_CONFIG;
};

// Save user configuration to localStorage
export const saveUserConfiguration = (config: UserConfiguration): void => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch (error) {
    console.error('Failed to save user configuration to localStorage:', error);
  }
};

// Validate configuration values
export const validateConfiguration = (config: Partial<UserConfiguration>): UserConfiguration => {
  return {
    rollingWindowSize: Math.max(1, Math.min(20, config.rollingWindowSize || DEFAULT_CONFIG.rollingWindowSize)),
    refreshRate: Math.max(5, Math.min(120, config.refreshRate || DEFAULT_CONFIG.refreshRate)),
    predictionConfidenceThreshold: Math.max(0.1, Math.min(1.0, config.predictionConfidenceThreshold || DEFAULT_CONFIG.predictionConfidenceThreshold)),
    chartUpdateInterval: Math.max(500, Math.min(5000, config.chartUpdateInterval || DEFAULT_CONFIG.chartUpdateInterval)),
    maxHistoryPoints: Math.max(10, Math.min(200, config.maxHistoryPoints || DEFAULT_CONFIG.maxHistoryPoints))
  };
};

// Reset configuration to defaults
export const resetConfiguration = (): UserConfiguration => {
  saveUserConfiguration(DEFAULT_CONFIG);
  return DEFAULT_CONFIG;
};