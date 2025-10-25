"""
Comprehensive error handling utilities for MomentumML.

This module provides centralized error handling, retry logic with exponential backoff,
logging utilities, and graceful degradation mechanisms.
"""

import asyncio
import functools
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from enum import Enum
import traceback

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error severity levels for categorization and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification and handling."""
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    DATA_PROCESSING_ERROR = "data_processing_error"
    ML_MODEL_ERROR = "ml_model_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_ERROR = "system_error"


class MomentumMLError(Exception):
    """Base exception class for MomentumML application errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.original_error = original_error
        self.timestamp = datetime.now(timezone.utc)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and API responses."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_error": str(self.original_error) if self.original_error else None
        }


class APIError(MomentumMLError):
    """Exception for NBA API related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.API_ERROR, **kwargs)
        self.status_code = status_code
        self.details["status_code"] = status_code


class DatabaseError(MomentumMLError):
    """Exception for database related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE_ERROR, **kwargs)


class DataProcessingError(MomentumMLError):
    """Exception for data processing and calculation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA_PROCESSING_ERROR, **kwargs)


class MLModelError(MomentumMLError):
    """Exception for machine learning model errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.ML_MODEL_ERROR, **kwargs)


class NetworkError(MomentumMLError):
    """Exception for network connectivity errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK_ERROR, **kwargs)


# Configure structured logging
def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> logging.Logger:
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_console: Whether to enable console logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("momentum_ml")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logging()


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> None:
    """
    Log an error with structured information.
    
    Args:
        error: Exception to log
        context: Additional context information
        severity: Error severity level
    """
    context = context or {}
    
    if isinstance(error, MomentumMLError):
        error_dict = error.to_dict()
        error_dict.update(context)
        
        # Rename 'message' to avoid conflict with logging's message field
        if 'message' in error_dict:
            error_dict['error_message'] = error_dict.pop('message')
        
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        logger.log(log_level, f"MomentumML Error: {error.message}", extra=error_dict)
    else:
        logger.error(
            f"Unexpected error: {str(error)}",
            extra={
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "context": context
            }
        )


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,)
) -> Callable[[F], F]:
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        log_error(
                            e,
                            context={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "args": str(args)[:200],  # Truncate for logging
                                "kwargs": str(kwargs)[:200]
                            },
                            severity=ErrorSeverity.HIGH
                        )
                        raise APIError(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            original_error=e,
                            severity=ErrorSeverity.HIGH
                        )
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    log_error(
                        e,
                        context={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "retry_delay": delay,
                            "max_retries": max_retries
                        },
                        severity=ErrorSeverity.LOW
                    )
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        log_error(
                            e,
                            context={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "args": str(args)[:200],
                                "kwargs": str(kwargs)[:200]
                            },
                            severity=ErrorSeverity.HIGH
                        )
                        raise APIError(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            original_error=e,
                            severity=ErrorSeverity.HIGH
                        )
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    log_error(
                        e,
                        context={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "retry_delay": delay,
                            "max_retries": max_retries
                        },
                        severity=ErrorSeverity.LOW
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_api_errors(func: F) -> F:
    """
    Decorator to handle and standardize API errors.
    
    Args:
        func: Function to wrap with API error handling
        
    Returns:
        Wrapped function with standardized error handling
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except APIError:
            raise  # Re-raise our custom API errors
        except Exception as e:
            # Convert generic exceptions to APIError
            error_message = f"API operation failed in {func.__name__}: {str(e)}"
            log_error(e, context={"function": func.__name__})
            raise APIError(error_message, original_error=e)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError:
            raise  # Re-raise our custom API errors
        except Exception as e:
            # Convert generic exceptions to APIError
            error_message = f"API operation failed in {func.__name__}: {str(e)}"
            log_error(e, context={"function": func.__name__})
            raise APIError(error_message, original_error=e)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class GracefulDegradation:
    """
    Utility class for implementing graceful degradation patterns.
    """
    
    def __init__(self):
        self.fallback_data: Dict[str, Any] = {}
        self.service_status: Dict[str, bool] = {}
    
    def set_fallback_data(self, key: str, data: Any) -> None:
        """Set fallback data for a specific key."""
        self.fallback_data[key] = {
            "data": data,
            "timestamp": datetime.now(timezone.utc)
        }
    
    def get_fallback_data(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """
        Get fallback data if it exists and is not too old.
        
        Args:
            key: Data key
            max_age_seconds: Maximum age of fallback data in seconds
            
        Returns:
            Fallback data if available and fresh, None otherwise
        """
        if key not in self.fallback_data:
            return None
        
        fallback = self.fallback_data[key]
        age = (datetime.now(timezone.utc) - fallback["timestamp"]).total_seconds()
        
        if age <= max_age_seconds:
            return fallback["data"]
        
        # Remove stale data
        del self.fallback_data[key]
        return None
    
    def set_service_status(self, service: str, is_healthy: bool) -> None:
        """Set the health status of a service."""
        self.service_status[service] = is_healthy
    
    def is_service_healthy(self, service: str) -> bool:
        """Check if a service is healthy."""
        return self.service_status.get(service, True)  # Default to healthy
    
    def with_fallback(
        self,
        primary_func: Callable[[], T],
        fallback_func: Optional[Callable[[], T]] = None,
        fallback_key: Optional[str] = None,
        cache_result: bool = True
    ) -> T:
        """
        Execute primary function with fallback to cached data or fallback function.
        
        Args:
            primary_func: Primary function to execute
            fallback_func: Optional fallback function
            fallback_key: Key for cached fallback data
            cache_result: Whether to cache successful results
            
        Returns:
            Result from primary function or fallback
        """
        try:
            result = primary_func()
            
            # Cache successful result if requested
            if cache_result and fallback_key:
                self.set_fallback_data(fallback_key, result)
            
            return result
            
        except Exception as e:
            log_error(
                e,
                context={
                    "primary_function": primary_func.__name__ if hasattr(primary_func, '__name__') else str(primary_func),
                    "fallback_key": fallback_key
                },
                severity=ErrorSeverity.MEDIUM
            )
            
            # Try fallback data first
            if fallback_key:
                fallback_data = self.get_fallback_data(fallback_key)
                if fallback_data is not None:
                    logger.warning(f"Using cached fallback data for {fallback_key}")
                    return fallback_data
            
            # Try fallback function
            if fallback_func:
                try:
                    logger.warning("Using fallback function")
                    return fallback_func()
                except Exception as fallback_error:
                    log_error(
                        fallback_error,
                        context={"fallback_function": fallback_func.__name__},
                        severity=ErrorSeverity.HIGH
                    )
            
            # No fallback available, re-raise original error
            raise e


# Global graceful degradation instance
graceful_degradation = GracefulDegradation()


def safe_execute(
    func: Callable[[], T],
    default_value: T,
    error_message: str = "Operation failed",
    log_errors: bool = True
) -> T:
    """
    Safely execute a function with a default return value on error.
    
    Args:
        func: Function to execute
        default_value: Value to return on error
        error_message: Custom error message for logging
        log_errors: Whether to log errors
        
    Returns:
        Function result or default value on error
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            log_error(
                e,
                context={
                    "function": func.__name__ if hasattr(func, '__name__') else str(func),
                    "error_message": error_message,
                    "default_value": str(default_value)
                },
                severity=ErrorSeverity.LOW
            )
        return default_value


def validate_data(
    data: Any,
    validation_func: Callable[[Any], bool],
    error_message: str = "Data validation failed"
) -> None:
    """
    Validate data and raise appropriate error if validation fails.
    
    Args:
        data: Data to validate
        validation_func: Function that returns True if data is valid
        error_message: Error message for validation failure
        
    Raises:
        DataProcessingError: If validation fails
    """
    try:
        if not validation_func(data):
            raise DataProcessingError(
                error_message,
                details={"data_type": type(data).__name__},
                severity=ErrorSeverity.MEDIUM
            )
    except Exception as e:
        if isinstance(e, DataProcessingError):
            raise
        raise DataProcessingError(
            f"Validation error: {error_message}",
            original_error=e,
            details={"data_type": type(data).__name__},
            severity=ErrorSeverity.MEDIUM
        )


# Health check utilities
class HealthChecker:
    """Utility class for service health monitoring."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.last_check_results: Dict[str, Tuple[bool, datetime]] = {}
    
    def register_health_check(self, service_name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function for a service."""
        self.health_checks[service_name] = check_func
    
    def check_service_health(self, service_name: str) -> bool:
        """Check the health of a specific service."""
        if service_name not in self.health_checks:
            return True  # Unknown services are assumed healthy
        
        try:
            is_healthy = self.health_checks[service_name]()
            self.last_check_results[service_name] = (is_healthy, datetime.now(timezone.utc))
            graceful_degradation.set_service_status(service_name, is_healthy)
            return is_healthy
        except Exception as e:
            log_error(
                e,
                context={"service": service_name, "check_type": "health_check"},
                severity=ErrorSeverity.MEDIUM
            )
            self.last_check_results[service_name] = (False, datetime.now(timezone.utc))
            graceful_degradation.set_service_status(service_name, False)
            return False
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_status = {}
        overall_healthy = True
        
        for service_name in self.health_checks:
            is_healthy = self.check_service_health(service_name)
            health_status[service_name] = {
                "healthy": is_healthy,
                "last_check": self.last_check_results.get(service_name, (None, None))[1]
            }
            if not is_healthy:
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "services": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global health checker instance
health_checker = HealthChecker()