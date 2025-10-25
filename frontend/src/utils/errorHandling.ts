/**
 * Comprehensive error handling utilities for MomentumML frontend.
 * 
 * This module provides centralized error handling, retry logic,
 * user-friendly error messages, and graceful degradation for the React application.
 */

import React from 'react';

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum ErrorCategory {
  NETWORK = 'network',
  API = 'api',
  WEBSOCKET = 'websocket',
  DATA_PROCESSING = 'data_processing',
  VALIDATION = 'validation',
  SYSTEM = 'system'
}

export class AppError extends Error {
  public readonly category: ErrorCategory;
  public readonly severity: ErrorSeverity;
  public readonly details: Record<string, any>;
  public readonly originalError?: Error;
  public readonly timestamp: Date;
  public readonly userMessage: string;
  public readonly retryable: boolean;

  constructor(
    message: string,
    category: ErrorCategory,
    severity: ErrorSeverity,
    options: {
      details?: Record<string, any>;
      originalError?: Error;
      userMessage?: string;
      retryable?: boolean;
    } = {}
  ) {
    super(message);
    this.name = 'AppError';
    this.category = category;
    this.severity = severity;
    this.details = options.details || {};
    this.originalError = options.originalError;
    this.timestamp = new Date();
    this.userMessage = options.userMessage || getUserFriendlyMessage(message, category);
    this.retryable = options.retryable ?? isRetryableError(category, message);
  }
}

export interface RetryOptions {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  backoffFactor: number;
}

export interface ErrorState {
  hasError: boolean;
  error: AppError | null;
  isRetrying: boolean;
  retryCount: number;
}

/**
 * Create a standardized application error
 */
export function createAppError(
  message: string,
  category: ErrorCategory = ErrorCategory.SYSTEM,
  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
  options: {
    details?: Record<string, any>;
    originalError?: Error;
    userMessage?: string;
    retryable?: boolean;
  } = {}
): AppError {
  return new AppError(message, category, severity, options);
}

/**
 * Generate user-friendly error messages
 */
export function getUserFriendlyMessage(message: string, category: ErrorCategory): string {
  const lowerMessage = message.toLowerCase();
  
  switch (category) {
    case ErrorCategory.NETWORK:
      if (lowerMessage.includes('timeout')) {
        return 'The request is taking longer than expected. Please check your connection and try again.';
      }
      if (lowerMessage.includes('offline') || lowerMessage.includes('network')) {
        return 'You appear to be offline. Please check your internet connection.';
      }
      return 'Unable to connect to the server. Please check your connection and try again.';
      
    case ErrorCategory.API:
      if (lowerMessage.includes('404') || lowerMessage.includes('not found')) {
        return 'The requested game or data was not found. It may no longer be available.';
      }
      if (lowerMessage.includes('503') || lowerMessage.includes('unavailable')) {
        return 'The service is temporarily unavailable. Please try again in a few moments.';
      }
      if (lowerMessage.includes('500') || lowerMessage.includes('internal server')) {
        return 'A server error occurred. Our team has been notified and is working on a fix.';
      }
      if (lowerMessage.includes('rate limit')) {
        return 'Too many requests. Please wait a moment before trying again.';
      }
      return 'Unable to fetch data from the server. Please try again.';
      
    case ErrorCategory.WEBSOCKET:
      if (lowerMessage.includes('connection')) {
        return 'Lost connection to live updates. Attempting to reconnect...';
      }
      return 'Real-time updates are temporarily unavailable. Data may be delayed.';
      
    case ErrorCategory.DATA_PROCESSING:
      return 'There was an issue processing the game data. Some information may be incomplete.';
      
    case ErrorCategory.VALIDATION:
      return 'Invalid data was provided. Please check your input and try again.';
      
    default:
      return 'An unexpected error occurred. Please try refreshing the page.';
  }
}

/**
 * Determine if an error is retryable
 */
export function isRetryableError(category: ErrorCategory, message: string): boolean {
  const lowerMessage = message.toLowerCase();
  
  // Network errors are generally retryable
  if (category === ErrorCategory.NETWORK) {
    return true;
  }
  
  // API errors - some are retryable
  if (category === ErrorCategory.API) {
    // Don't retry client errors (4xx except 408, 429)
    if (lowerMessage.includes('400') || lowerMessage.includes('401') || 
        lowerMessage.includes('403') || lowerMessage.includes('404')) {
      return false;
    }
    // Retry server errors and specific client errors
    return lowerMessage.includes('500') || lowerMessage.includes('502') || 
           lowerMessage.includes('503') || lowerMessage.includes('504') ||
           lowerMessage.includes('408') || lowerMessage.includes('429');
  }
  
  // WebSocket errors are retryable
  if (category === ErrorCategory.WEBSOCKET) {
    return true;
  }
  
  // Data processing and validation errors are not retryable
  return false;
}

/**
 * Retry function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
    backoffFactor: 2
  }
): Promise<T> {
  let lastError: Error;
  
  for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      
      if (attempt === options.maxRetries) {
        throw lastError;
      }
      
      // Calculate delay with exponential backoff
      const delay = Math.min(
        options.baseDelay * Math.pow(options.backoffFactor, attempt),
        options.maxDelay
      );
      
      console.warn(`Attempt ${attempt + 1} failed, retrying in ${delay}ms:`, error);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError!;
}

/**
 * Fetch with error handling and retry logic
 */
export async function fetchWithErrorHandling(
  url: string,
  options: RequestInit = {},
  retryOptions?: Partial<RetryOptions>
): Promise<Response> {
  const fullRetryOptions: RetryOptions = {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
    backoffFactor: 2,
    ...retryOptions
  };
  
  return retryWithBackoff(async () => {
    try {
      const response = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });
      
      if (!response.ok) {
        const errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        throw createAppError(
          errorMessage,
          ErrorCategory.API,
          response.status >= 500 ? ErrorSeverity.HIGH : ErrorSeverity.MEDIUM,
          {
            details: {
              status: response.status,
              statusText: response.statusText,
              url
            },
            retryable: response.status >= 500 || response.status === 408 || response.status === 429
          }
        );
      }
      
      return response;
    } catch (error) {
      if (error instanceof AppError) {
        throw error;
      }
      
      // Handle network errors
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw createAppError(
          'Network request failed',
          ErrorCategory.NETWORK,
          ErrorSeverity.HIGH,
          {
            originalError: error as Error,
            details: { url }
          }
        );
      }
      
      // Handle timeout errors
      if (error instanceof DOMException && error.name === 'TimeoutError') {
        throw createAppError(
          'Request timeout',
          ErrorCategory.NETWORK,
          ErrorSeverity.MEDIUM,
          {
            originalError: error as Error,
            details: { url }
          }
        );
      }
      
      throw createAppError(
        'Unexpected fetch error',
        ErrorCategory.SYSTEM,
        ErrorSeverity.HIGH,
        {
          originalError: error as Error,
          details: { url }
        }
      );
    }
  }, fullRetryOptions);
}

/**
 * WebSocket error handler
 */
export function handleWebSocketError(error: Event): AppError {
  return createAppError(
    'WebSocket connection error',
    ErrorCategory.WEBSOCKET,
    ErrorSeverity.MEDIUM,
    {
      details: {
        type: error.type,
        timestamp: new Date().toISOString()
      },
      userMessage: 'Lost connection to live updates. Attempting to reconnect...',
      retryable: true
    }
  );
}

/**
 * Log error for debugging and monitoring
 */
export function logError(error: AppError, context?: Record<string, any>): void {
  const logData = {
    message: error.message,
    category: error.category,
    severity: error.severity,
    timestamp: error.timestamp.toISOString(),
    details: error.details,
    context,
    userAgent: navigator.userAgent,
    url: window.location.href
  };
  
  // Log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.group(`🚨 ${error.severity.toUpperCase()} Error - ${error.category}`);
    console.error('Message:', error.message);
    console.error('User Message:', error.userMessage);
    console.error('Details:', error.details);
    if (error.originalError) {
      console.error('Original Error:', error.originalError);
    }
    if (context) {
      console.error('Context:', context);
    }
    console.groupEnd();
  }
  
  // In production, you would send this to your error tracking service
  // Example: Sentry, LogRocket, etc.
  if (process.env.NODE_ENV === 'production') {
    // sendToErrorTrackingService(logData);
  }
}

/**
 * Error boundary helper for React components
 */
export function createErrorBoundaryError(error: Error, errorInfo: any): AppError {
  return createAppError(
    'Component error occurred',
    ErrorCategory.SYSTEM,
    ErrorSeverity.HIGH,
    {
      originalError: error,
      details: {
        componentStack: errorInfo.componentStack,
        errorBoundary: true
      },
      userMessage: 'Something went wrong with this component. Please try refreshing the page.',
      retryable: false
    }
  );
}

/**
 * Graceful degradation utility
 */
export class GracefulDegradation {
  private fallbackData: Map<string, { data: any; timestamp: Date }> = new Map();
  private serviceStatus: Map<string, boolean> = new Map();
  
  setFallbackData(key: string, data: any): void {
    this.fallbackData.set(key, {
      data,
      timestamp: new Date()
    });
  }
  
  getFallbackData(key: string, maxAgeMs: number = 300000): any | null {
    const fallback = this.fallbackData.get(key);
    if (!fallback) return null;
    
    const age = Date.now() - fallback.timestamp.getTime();
    if (age <= maxAgeMs) {
      return fallback.data;
    }
    
    // Remove stale data
    this.fallbackData.delete(key);
    return null;
  }
  
  setServiceStatus(service: string, isHealthy: boolean): void {
    this.serviceStatus.set(service, isHealthy);
  }
  
  isServiceHealthy(service: string): boolean {
    return this.serviceStatus.get(service) ?? true;
  }
  
  withFallback<T>(
    primaryFn: () => Promise<T>,
    fallbackKey?: string,
    fallbackFn?: () => T
  ): Promise<T> {
    return primaryFn().catch((error) => {
      logError(error instanceof AppError ? error : createAppError(
        'Primary function failed',
        ErrorCategory.SYSTEM,
        ErrorSeverity.MEDIUM,
        { originalError: error }
      ));
      
      // Try fallback data first
      if (fallbackKey) {
        const fallbackData = this.getFallbackData(fallbackKey);
        if (fallbackData !== null) {
          console.warn(`Using fallback data for ${fallbackKey}`);
          return fallbackData;
        }
      }
      
      // Try fallback function
      if (fallbackFn) {
        try {
          console.warn('Using fallback function');
          return Promise.resolve(fallbackFn());
        } catch (fallbackError) {
          logError(createAppError(
            'Fallback function failed',
            ErrorCategory.SYSTEM,
            ErrorSeverity.HIGH,
            { originalError: fallbackError as Error }
          ));
        }
      }
      
      // No fallback available, re-throw original error
      throw error;
    });
  }
}

// Global instance
export const gracefulDegradation = new GracefulDegradation();

/**
 * React hook for error state management
 */
export function useErrorState() {
  const [errorState, setErrorState] = React.useState<ErrorState>({
    hasError: false,
    error: null,
    isRetrying: false,
    retryCount: 0
  });
  
  const setError = React.useCallback((error: AppError) => {
    setErrorState(prev => ({
      hasError: true,
      error,
      isRetrying: false,
      retryCount: prev.retryCount
    }));
    logError(error);
  }, []);
  
  const clearError = React.useCallback(() => {
    setErrorState({
      hasError: false,
      error: null,
      isRetrying: false,
      retryCount: 0
    });
  }, []);
  
  const retry = React.useCallback(async (retryFn: () => Promise<void>) => {
    if (!errorState.error?.retryable) {
      return;
    }
    
    setErrorState(prev => ({
      ...prev,
      isRetrying: true,
      retryCount: prev.retryCount + 1
    }));
    
    try {
      await retryFn();
      clearError();
    } catch (error) {
      const appError = error instanceof AppError ? error : createAppError(
        'Retry failed',
        ErrorCategory.SYSTEM,
        ErrorSeverity.MEDIUM,
        { originalError: error as Error }
      );
      setErrorState(prev => ({
        hasError: true,
        error: appError,
        isRetrying: false,
        retryCount: prev.retryCount + 1
      }));
    }
  }, [errorState.error, clearError]);
  
  return {
    errorState,
    setError,
    clearError,
    retry
  };
}