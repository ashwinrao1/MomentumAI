# MomentumML Error Handling and Resilience Implementation

## Overview

This document describes the comprehensive error handling and resilience features implemented in the MomentumML application. The implementation covers both backend (Python/FastAPI) and frontend (React/TypeScript) components with centralized error management, retry logic, graceful degradation, and user-friendly error reporting.

## Backend Error Handling

### Core Components

#### 1. Error Handling Utilities (`backend/utils/error_handling.py`)

**Custom Exception Classes:**
- `MomentumMLError`: Base exception class with categorization and severity levels
- `APIError`: NBA API related errors with status code tracking
- `DatabaseError`: Database operation errors
- `DataProcessingError`: Data validation and processing errors
- `MLModelError`: Machine learning model errors
- `NetworkError`: Network connectivity errors

**Error Categorization:**
```python
class ErrorCategory(Enum):
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    DATA_PROCESSING_ERROR = "data_processing_error"
    ML_MODEL_ERROR = "ml_model_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_ERROR = "system_error"
```

**Severity Levels:**
```python
class ErrorSeverity(Enum):
    LOW = "low"          # Minor issues, system continues normally
    MEDIUM = "medium"    # Moderate issues, some functionality affected
    HIGH = "high"        # Serious issues, major functionality impacted
    CRITICAL = "critical" # System-threatening issues
```

#### 2. Retry Logic with Exponential Backoff

**Decorator Usage:**
```python
@retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    exceptions=(APIError, NetworkError)
)
async def fetch_data():
    # Function implementation
    pass
```

**Features:**
- Configurable retry attempts and delays
- Exponential backoff with jitter
- Exception type filtering
- Automatic logging of retry attempts

#### 3. Graceful Degradation

**Fallback Data Management:**
```python
# Store fallback data
graceful_degradation.set_fallback_data("active_games", games_data)

# Retrieve with age limit
cached_games = graceful_degradation.get_fallback_data("active_games", max_age_seconds=300)

# Use with fallback function
result = graceful_degradation.with_fallback(
    primary_func=fetch_live_data,
    fallback_key="cached_data",
    fallback_func=get_default_data
)
```

#### 4. Health Monitoring

**Service Health Checks:**
```python
# Register health check
health_checker.register_health_check("nba_api", check_nba_api_health)

# Check individual service
is_healthy = health_checker.check_service_health("nba_api")

# Get overall system health
health_status = health_checker.get_overall_health()
```

#### 5. Structured Logging

**Log Configuration:**
```python
logger = setup_logging(
    log_level="INFO",
    log_file="momentum_ml.log",
    enable_console=True
)
```

**Error Logging:**
```python
log_error(
    error,
    context={"game_id": game_id, "operation": "fetch_data"},
    severity=ErrorSeverity.HIGH
)
```

### Service-Level Implementation

#### 1. Live Data Fetcher (`backend/services/live_fetcher.py`)

**Enhanced Features:**
- Retry logic for NBA API calls
- Rate limiting with exponential backoff
- Data validation and error handling
- Fallback to cached data on API failures
- Health check registration

**Example Implementation:**
```python
@retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
@handle_api_errors
async def fetch_live_game_data(self, game_id: str):
    try:
        # Validate input
        validate_data(game_id, lambda gid: isinstance(gid, str) and len(gid) > 0)
        
        # Check service health
        if not graceful_degradation.is_service_healthy("nba_api"):
            return self._get_cached_data(game_id)
        
        # Fetch data with error handling
        # ... implementation
        
    except APIError as e:
        # Try fallback data
        cached_data = self._get_cached_data(game_id)
        if cached_data[0] is not None:
            return cached_data
        raise
```

#### 2. Momentum Engine (`backend/services/momentum_engine.py`)

**Error Handling Features:**
- Input validation for all calculations
- Safe execution of mathematical operations
- Graceful handling of missing or invalid data
- ML model error recovery
- Feature calculation error isolation

#### 3. API Endpoints (`backend/api/momentum_endpoints.py`)

**Enhanced Error Responses:**
- Standardized error response format
- Appropriate HTTP status codes
- User-friendly error messages
- Detailed error information for debugging
- Service health status integration

**Example Error Response:**
```json
{
  "error": {
    "message": "Failed to fetch game data",
    "category": "api_error",
    "severity": "high",
    "timestamp": "2025-10-23T21:00:00Z",
    "details": {
      "game_id": "0022500123",
      "status_code": 503
    }
  }
}
```

### Global Exception Handling

**FastAPI Exception Handlers:**
```python
@app.exception_handler(MomentumMLError)
async def momentum_ml_exception_handler(request: Request, exc: MomentumMLError):
    # Handle custom exceptions with appropriate status codes
    pass

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Handle unexpected exceptions with logging
    pass
```

## Frontend Error Handling

### Core Components

#### 1. Error Handling Utilities (`frontend/src/utils/errorHandling.ts`)

**Error Types:**
```typescript
interface AppError {
  message: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  details?: Record<string, any>;
  originalError?: Error;
  timestamp: Date;
  userMessage: string;
  retryable: boolean;
}
```

**Key Features:**
- User-friendly error message generation
- Retry logic with exponential backoff
- Fetch wrapper with error handling
- Graceful degradation utilities
- Error state management hook

#### 2. Enhanced WebSocket Hook (`frontend/src/hooks/useWebSocket.ts`)

**Features:**
- Automatic reconnection with exponential backoff
- Connection health monitoring
- Heartbeat mechanism
- Comprehensive error handling
- Service status integration

**Usage:**
```typescript
const {
  connectionStatus,
  lastMessage,
  connect,
  disconnect,
  error,
  isHealthy
} = useWebSocket({
  url: 'ws://localhost:8000/live/stream',
  onMessage: handleMessage,
  onError: handleError,
  maxReconnectAttempts: 10
});
```

#### 3. Error Display Component (`frontend/src/components/ErrorDisplay/ErrorDisplay.tsx`)

**Features:**
- Severity-based styling and icons
- Retry functionality for retryable errors
- Expandable technical details
- Dismissible notifications
- Accessibility compliance

#### 4. Enhanced Dashboard (`frontend/src/components/Dashboard/Dashboard.tsx`)

**Error Handling Integration:**
- Comprehensive error state management
- Fallback data usage
- Loading states and error displays
- Service health indicators
- Automatic error recovery

### Error Recovery Strategies

#### 1. Network Errors
- Automatic retry with exponential backoff
- Fallback to cached data
- User notification with retry option
- Connection status indicators

#### 2. API Errors
- Differentiate between client and server errors
- Retry server errors, don't retry client errors
- Graceful degradation to cached data
- User-friendly error messages

#### 3. WebSocket Errors
- Automatic reconnection attempts
- Heartbeat monitoring
- Connection health indicators
- Fallback to polling if needed

#### 4. Data Processing Errors
- Input validation and sanitization
- Safe execution with default values
- Error isolation to prevent cascading failures
- Partial data display when possible

## Configuration and Monitoring

### Environment Variables

```bash
# API Configuration
NBA_API_BASE_URL=https://stats.nba.com
API_POLL_INTERVAL=25
API_TIMEOUT=30
API_MAX_RETRIES=3

# Error Handling
LOG_LEVEL=INFO
LOG_FILE=momentum_ml.log
ENABLE_ERROR_TRACKING=true

# Health Monitoring
HEALTH_CHECK_INTERVAL=60
SERVICE_TIMEOUT=10
```

### Health Check Endpoints

**Backend Health Check:**
```
GET /health
GET /api/momentum/health
GET /api/momentum/status
```

**Response Format:**
```json
{
  "overall_healthy": true,
  "services": {
    "nba_api": {
      "healthy": true,
      "last_check": "2025-10-23T21:00:00Z"
    },
    "momentum_engine": {
      "healthy": true,
      "last_check": "2025-10-23T21:00:00Z"
    },
    "database": {
      "healthy": true,
      "last_check": "2025-10-23T21:00:00Z"
    }
  },
  "timestamp": "2025-10-23T21:00:00Z"
}
```

### Monitoring and Alerting

**Logging Integration:**
- Structured logging with JSON format
- Error categorization and severity levels
- Context information for debugging
- Integration with external monitoring services

**Metrics Collection:**
- Error rates by category and severity
- Service health status
- API response times
- Retry attempt statistics

## Testing

### Backend Tests

**Error Handling Test Suite:**
```bash
python backend/test_error_handling.py
```

**Test Coverage:**
- Custom error creation and logging
- Retry logic with exponential backoff
- Graceful degradation functionality
- Safe execution utilities
- Data validation
- Health checker functionality

### Frontend Tests

**Jest Test Suite:**
```bash
npm test -- errorHandling.test.ts
```

**Test Coverage:**
- Error creation and categorization
- User-friendly message generation
- Retry logic determination
- Graceful degradation utilities
- WebSocket error handling

## Best Practices

### Error Handling Guidelines

1. **Always use custom error types** for better categorization and handling
2. **Implement retry logic** for transient failures (network, API timeouts)
3. **Provide fallback data** when possible to maintain functionality
4. **Log errors with context** for effective debugging
5. **Use appropriate HTTP status codes** in API responses
6. **Display user-friendly messages** while preserving technical details
7. **Implement health checks** for all critical services
8. **Monitor error rates and patterns** for proactive issue resolution

### Performance Considerations

1. **Limit retry attempts** to prevent resource exhaustion
2. **Use exponential backoff** to avoid overwhelming failing services
3. **Cache fallback data** with appropriate expiration times
4. **Implement circuit breakers** for failing external services
5. **Monitor memory usage** of error logging and caching

### Security Considerations

1. **Sanitize error messages** to prevent information leakage
2. **Log security-relevant errors** at appropriate levels
3. **Implement rate limiting** for error-prone endpoints
4. **Validate all inputs** to prevent injection attacks
5. **Use secure logging practices** for sensitive information

## Deployment and Operations

### Production Configuration

1. **Enable structured logging** with appropriate log levels
2. **Configure external monitoring** services (e.g., Sentry, DataDog)
3. **Set up alerting** for critical errors and service failures
4. **Implement log rotation** to manage disk space
5. **Configure health check endpoints** for load balancers

### Troubleshooting Guide

**Common Issues:**
1. **High error rates** - Check service health and external dependencies
2. **Memory leaks** - Review error caching and logging retention
3. **Performance degradation** - Analyze retry patterns and fallback usage
4. **User complaints** - Review error messages and recovery mechanisms

**Debugging Steps:**
1. Check application logs for error patterns
2. Verify service health status
3. Test external API connectivity
4. Review error categorization and severity
5. Validate fallback data availability

## Future Enhancements

### Planned Improvements

1. **Circuit breaker pattern** implementation
2. **Advanced retry strategies** (jittered, adaptive)
3. **Error analytics dashboard** for operations team
4. **Automated error recovery** for common failure scenarios
5. **Integration with external monitoring** services
6. **Performance metrics collection** and analysis
7. **A/B testing** for error handling strategies

### Monitoring Enhancements

1. **Real-time error dashboards**
2. **Predictive failure detection**
3. **Automated incident response**
4. **Error trend analysis**
5. **Service dependency mapping**

This comprehensive error handling implementation ensures that MomentumML provides a robust, resilient, and user-friendly experience even when facing various types of failures and issues.