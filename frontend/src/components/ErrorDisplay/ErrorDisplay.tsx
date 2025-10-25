import React from 'react';
import { AppError, ErrorSeverity, ErrorCategory } from '../../utils/errorHandling';

interface ErrorDisplayProps {
  error: AppError;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
  showDetails?: boolean;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  onRetry,
  onDismiss,
  className = '',
  showDetails = false
}) => {
  const [showFullDetails, setShowFullDetails] = React.useState(false);

  const getSeverityStyles = (severity: ErrorSeverity) => {
    switch (severity) {
      case ErrorSeverity.LOW:
        return {
          container: 'bg-blue-50 border-blue-200 text-blue-800',
          icon: 'text-blue-400',
          button: 'bg-blue-100 hover:bg-blue-200 text-blue-800'
        };
      case ErrorSeverity.MEDIUM:
        return {
          container: 'bg-yellow-50 border-yellow-200 text-yellow-800',
          icon: 'text-yellow-400',
          button: 'bg-yellow-100 hover:bg-yellow-200 text-yellow-800'
        };
      case ErrorSeverity.HIGH:
        return {
          container: 'bg-red-50 border-red-200 text-red-800',
          icon: 'text-red-400',
          button: 'bg-red-100 hover:bg-red-200 text-red-800'
        };
      case ErrorSeverity.CRITICAL:
        return {
          container: 'bg-red-100 border-red-300 text-red-900',
          icon: 'text-red-500',
          button: 'bg-red-200 hover:bg-red-300 text-red-900'
        };
      default:
        return {
          container: 'bg-gray-50 border-gray-200 text-gray-800',
          icon: 'text-gray-400',
          button: 'bg-gray-100 hover:bg-gray-200 text-gray-800'
        };
    }
  };

  const getIcon = (category: ErrorCategory, severity: ErrorSeverity) => {
    const iconClass = `w-5 h-5 ${getSeverityStyles(severity).icon}`;
    
    switch (category) {
      case ErrorCategory.NETWORK:
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case ErrorCategory.API:
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case ErrorCategory.WEBSOCKET:
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        );
      case ErrorCategory.DATA_PROCESSING:
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        );
      default:
        return (
          <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  const styles = getSeverityStyles(error.severity);

  return (
    <div className={`border rounded-lg p-4 ${styles.container} ${className}`}>
      <div className="flex items-start">
        <div className="flex-shrink-0">
          {getIcon(error.category, error.severity)}
        </div>
        
        <div className="ml-3 flex-1">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">
              {error.severity === ErrorSeverity.CRITICAL ? 'Critical Error' :
               error.severity === ErrorSeverity.HIGH ? 'Error' :
               error.severity === ErrorSeverity.MEDIUM ? 'Warning' : 'Notice'}
            </h3>
            
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="ml-2 flex-shrink-0 rounded-md p-1.5 hover:bg-black hover:bg-opacity-10 focus:outline-none focus:ring-2 focus:ring-offset-2"
                aria-label="Dismiss error"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          
          <div className="mt-2 text-sm">
            <p>{error.userMessage}</p>
          </div>
          
          {(showDetails || showFullDetails) && (
            <div className="mt-3 text-xs">
              <details className="cursor-pointer">
                <summary className="font-medium hover:underline">
                  Technical Details
                </summary>
                <div className="mt-2 space-y-1">
                  <p><strong>Category:</strong> {error.category}</p>
                  <p><strong>Severity:</strong> {error.severity}</p>
                  <p><strong>Time:</strong> {error.timestamp.toLocaleString()}</p>
                  {error.message !== error.userMessage && (
                    <p><strong>Technical Message:</strong> {error.message}</p>
                  )}
                  {error.details && Object.keys(error.details).length > 0 && (
                    <div>
                      <strong>Details:</strong>
                      <pre className="mt-1 text-xs bg-black bg-opacity-10 p-2 rounded overflow-x-auto">
                        {JSON.stringify(error.details, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </details>
            </div>
          )}
          
          {(error.retryable || onRetry) && (
            <div className="mt-4 flex space-x-2">
              {error.retryable && onRetry && (
                <button
                  onClick={onRetry}
                  className={`inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md ${styles.button} focus:outline-none focus:ring-2 focus:ring-offset-2`}
                >
                  <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Try Again
                </button>
              )}
              
              {!showDetails && (
                <button
                  onClick={() => setShowFullDetails(!showFullDetails)}
                  className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-800 focus:outline-none"
                >
                  {showFullDetails ? 'Hide Details' : 'Show Details'}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorDisplay;