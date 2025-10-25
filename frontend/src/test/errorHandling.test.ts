/**
 * Test suite for frontend error handling utilities.
 */

import {
  createAppError,
  ErrorCategory,
  ErrorSeverity,
  getUserFriendlyMessage,
  isRetryableError,
  gracefulDegradation
} from '../utils/errorHandling';

describe('Error Handling Utilities', () => {
  describe('createAppError', () => {
    it('should create a basic error with defaults', () => {
      const error = createAppError('Test error');
      
      expect(error.message).toBe('Test error');
      expect(error.category).toBe(ErrorCategory.SYSTEM);
      expect(error.severity).toBe(ErrorSeverity.MEDIUM);
      expect(error.timestamp).toBeInstanceOf(Date);
      expect(error.userMessage).toBeTruthy();
      expect(typeof error.retryable).toBe('boolean');
    });
    
    it('should create an error with custom options', () => {
      const originalError = new Error('Original error');
      const error = createAppError(
        'Custom error',
        ErrorCategory.API,
        ErrorSeverity.HIGH,
        {
          details: { code: 500 },
          originalError,
          userMessage: 'Custom user message',
          retryable: true
        }
      );
      
      expect(error.message).toBe('Custom error');
      expect(error.category).toBe(ErrorCategory.API);
      expect(error.severity).toBe(ErrorSeverity.HIGH);
      expect(error.details).toEqual({ code: 500 });
      expect(error.originalError).toBe(originalError);
      expect(error.userMessage).toBe('Custom user message');
      expect(error.retryable).toBe(true);
    });
  });
  
  describe('getUserFriendlyMessage', () => {
    it('should return appropriate message for network errors', () => {
      const message = getUserFriendlyMessage('timeout error', ErrorCategory.NETWORK);
      expect(message).toContain('longer than expected');
    });
    
    it('should return appropriate message for API errors', () => {
      const message = getUserFriendlyMessage('404 not found', ErrorCategory.API);
      expect(message).toContain('not found');
    });
    
    it('should return appropriate message for WebSocket errors', () => {
      const message = getUserFriendlyMessage('connection failed', ErrorCategory.WEBSOCKET);
      expect(message).toContain('connection');
    });
  });
  
  describe('isRetryableError', () => {
    it('should identify retryable network errors', () => {
      expect(isRetryableError(ErrorCategory.NETWORK, 'timeout')).toBe(true);
    });
    
    it('should identify non-retryable client errors', () => {
      expect(isRetryableError(ErrorCategory.API, '404 not found')).toBe(false);
      expect(isRetryableError(ErrorCategory.API, '400 bad request')).toBe(false);
    });
    
    it('should identify retryable server errors', () => {
      expect(isRetryableError(ErrorCategory.API, '500 internal server error')).toBe(true);
      expect(isRetryableError(ErrorCategory.API, '503 service unavailable')).toBe(true);
    });
    
    it('should identify non-retryable validation errors', () => {
      expect(isRetryableError(ErrorCategory.VALIDATION, 'invalid input')).toBe(false);
    });
  });
  
  describe('GracefulDegradation', () => {
    beforeEach(() => {
      // Clear any existing data
      gracefulDegradation['fallbackData'].clear();
      gracefulDegradation['serviceStatus'].clear();
    });
    
    it('should store and retrieve fallback data', () => {
      const testData = { test: 'value' };
      gracefulDegradation.setFallbackData('test-key', testData);
      
      const retrieved = gracefulDegradation.getFallbackData('test-key');
      expect(retrieved).toEqual(testData);
    });
    
    it('should return null for expired fallback data', () => {
      const testData = { test: 'value' };
      gracefulDegradation.setFallbackData('test-key', testData);
      
      // Try to get data with 0ms max age (should be expired)
      const retrieved = gracefulDegradation.getFallbackData('test-key', 0);
      expect(retrieved).toBeNull();
    });
    
    it('should track service status', () => {
      gracefulDegradation.setServiceStatus('test-service', false);
      expect(gracefulDegradation.isServiceHealthy('test-service')).toBe(false);
      
      gracefulDegradation.setServiceStatus('test-service', true);
      expect(gracefulDegradation.isServiceHealthy('test-service')).toBe(true);
    });
    
    it('should default to healthy for unknown services', () => {
      expect(gracefulDegradation.isServiceHealthy('unknown-service')).toBe(true);
    });
  });
});

// Mock console methods for testing
const originalConsole = { ...console };
beforeAll(() => {
  console.warn = jest.fn();
  console.error = jest.fn();
  console.log = jest.fn();
});

afterAll(() => {
  Object.assign(console, originalConsole);
});