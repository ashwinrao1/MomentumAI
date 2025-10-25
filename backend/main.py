from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
import asyncio
import time
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.game_endpoints import router as game_router
from backend.api.momentum_endpoints import router as momentum_router
from backend.api.websocket_endpoints import router as websocket_router, periodic_momentum_updates
from backend.api.performance_endpoints import router as performance_router
from backend.database.migrations import run_migrations
from backend.utils.error_handling import (
    setup_logging, log_error, ErrorSeverity, 
    MomentumMLError, APIError, DatabaseError,
    health_checker
)
from backend.utils.performance_monitor import get_metrics_collector

app = FastAPI(
    title="MomentumML API",
    description="Real-time basketball momentum analytics platform",
    version="1.0.0"
)

# Configure structured logging
logger = setup_logging(
    log_level="INFO",
    log_file="momentum_ml.log",
    enable_console=True
)


@app.on_event("startup")
async def startup_event():
    """Initialize database and services on application startup."""
    logger.info("Starting MomentumML API...")
    try:
        # Initialize database
        run_migrations()
        logger.info("Database initialization completed")
        
        # Register database health check
        def check_database_health():
            try:
                from backend.database.service import DatabaseService
                db_service = DatabaseService()
                return db_service.test_connection()
            except Exception:
                return False
        
        health_checker.register_health_check("database", check_database_health)
        
        # Run performance optimizations
        try:
            from backend.database.performance_migration import run_performance_migration
            run_performance_migration()
            logger.info("Performance optimizations applied")
        except Exception as e:
            logger.warning(f"Performance optimization failed: {e}")
        
        # Start background task for periodic momentum updates
        asyncio.create_task(periodic_momentum_updates())
        logger.info("Started periodic momentum updates background task")
        
        logger.info("MomentumML API startup completed successfully")
        
    except Exception as e:
        log_error(
            DatabaseError(f"Database initialization failed: {str(e)}", original_error=e),
            context={"startup": True},
            severity=ErrorSeverity.CRITICAL
        )
        raise


# Global exception handlers
@app.exception_handler(MomentumMLError)
async def momentum_ml_exception_handler(request: Request, exc: MomentumMLError):
    """Handle custom MomentumML exceptions."""
    log_error(exc, context={"url": str(request.url), "method": request.method})
    
    status_code = {
        ErrorSeverity.LOW: status.HTTP_200_OK,
        ErrorSeverity.MEDIUM: status.HTTP_400_BAD_REQUEST,
        ErrorSeverity.HIGH: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorSeverity.CRITICAL: status.HTTP_503_SERVICE_UNAVAILABLE
    }.get(exc.severity, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": exc.message,
                "category": exc.category.value,
                "severity": exc.severity.value,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error for {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Request validation failed",
                "category": "validation",
                "severity": "medium",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    log_error(
        exc,
        context={
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers)
        },
        severity=ErrorSeverity.CRITICAL
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "category": "system",
                "severity": "critical",
                "timestamp": exc.__class__.__name__
            }
        }
    )

# Performance monitoring middleware
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Middleware to track request performance metrics."""
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Get metrics collector
    metrics = get_metrics_collector()
    
    # Extract endpoint from path
    endpoint = request.url.path
    
    # Start tracking request
    metrics.start_request(request_id, endpoint)
    
    # Record request start time
    start_time = time.time()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Record successful request
        success = 200 <= response.status_code < 400
        metrics.end_request(request_id, endpoint, success)
        
        # Add performance headers
        duration_ms = (time.time() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        # Record failed request
        metrics.end_request(request_id, endpoint, False)
        
        # Re-raise the exception
        raise


# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(game_router)
app.include_router(momentum_router)
app.include_router(websocket_router)
app.include_router(performance_router)

@app.get("/")
async def root():
    return {
        "message": "MomentumML API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = health_checker.get_overall_health()
        
        http_status = status.HTTP_200_OK if health_status["overall_healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=http_status,
            content=health_status
        )
    except Exception as e:
        log_error(e, context={"endpoint": "health"}, severity=ErrorSeverity.HIGH)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "overall_healthy": False,
                "error": "Health check failed",
                "timestamp": "unknown"
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)