"""
Performance monitoring and metrics API endpoints.

This module provides REST endpoints for accessing performance metrics,
cache statistics, and system health information.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.utils.performance_monitor import get_metrics_collector
from backend.utils.cache import get_cache
from backend.utils.error_handling import log_error, ErrorSeverity

logger = logging.getLogger("momentum_ml.api.performance")

# Create router
router = APIRouter(prefix="/api/performance", tags=["performance"])


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    time_window_minutes: int
    generated_at: str
    timing_stats: Dict[str, Any]
    request_stats: Dict[str, Any]
    database_stats: Dict[str, Any]
    websocket_stats: Dict[str, Any]
    system_stats: Dict[str, Any]
    counter_metrics: Dict[str, Any]
    gauge_metrics: Dict[str, Any]
    active_requests: int


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    game_data_cache: Dict[str, Any]
    momentum_cache: Dict[str, Any]
    api_response_cache: Dict[str, Any]
    ml_prediction_cache: Dict[str, Any]
    total_memory_mb: float


class PerformanceAlertResponse(BaseModel):
    """Response model for performance alerts."""
    type: str
    severity: str
    message: str
    value: float
    threshold: float
    endpoint: Optional[str] = None


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    time_window_minutes: int = Query(5, description="Time window for metrics in minutes", ge=1, le=60)
):
    """
    Get comprehensive performance metrics for the specified time window.
    
    Args:
        time_window_minutes: Time window for metrics calculation (1-60 minutes)
        
    Returns:
        Comprehensive performance metrics including timing, requests, database, and system stats
    """
    try:
        metrics_collector = get_metrics_collector()
        metrics_summary = metrics_collector.get_metrics_summary(time_window_minutes)
        
        return PerformanceMetricsResponse(**metrics_summary)
        
    except Exception as e:
        log_error(e, context={"endpoint": "performance_metrics"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_statistics():
    """
    Get comprehensive cache statistics including hit rates and memory usage.
    
    Returns:
        Cache statistics for all cache layers
    """
    try:
        cache = get_cache()
        cache_stats = cache.get_cache_stats()
        
        return CacheStatsResponse(**cache_stats)
        
    except Exception as e:
        log_error(e, context={"endpoint": "cache_stats"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )


@router.get("/alerts")
async def get_performance_alerts():
    """
    Get current performance alerts based on predefined thresholds.
    
    Returns:
        List of active performance alerts
    """
    try:
        metrics_collector = get_metrics_collector()
        alerts = metrics_collector.get_performance_alerts()
        
        return JSONResponse(content={
            "alerts": alerts,
            "alert_count": len(alerts),
            "generated_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        log_error(e, context={"endpoint": "performance_alerts"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance alerts"
        )


@router.get("/system")
async def get_system_metrics():
    """
    Get current system resource usage metrics.
    
    Returns:
        Current system metrics including CPU, memory, and disk usage
    """
    try:
        metrics_collector = get_metrics_collector()
        
        # Force collection of current system metrics
        metrics_collector.collect_system_metrics()
        
        # Get the latest system metrics
        with metrics_collector._lock:
            if metrics_collector._system_metrics_history:
                latest_metrics = metrics_collector._system_metrics_history[-1]
                return JSONResponse(content={
                    **{k: v for k, v in latest_metrics.items() if k != 'timestamp'},
                    "collected_at": latest_metrics['timestamp'].isoformat()
                })
            else:
                return JSONResponse(content={
                    "error": "No system metrics available",
                    "collected_at": datetime.utcnow().isoformat()
                })
        
    except Exception as e:
        log_error(e, context={"endpoint": "system_metrics"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics"
        )


@router.get("/database")
async def get_database_performance():
    """
    Get database performance metrics including query times and connection pool stats.
    
    Returns:
        Database performance metrics
    """
    try:
        metrics_collector = get_metrics_collector()
        
        with metrics_collector._lock:
            # Get recent database query metrics
            recent_queries = list(metrics_collector._db_query_times)[-100:]  # Last 100 queries
            
            if recent_queries:
                avg_query_time = sum(q['duration_ms'] for q in recent_queries) / len(recent_queries)
                max_query_time = max(q['duration_ms'] for q in recent_queries)
                min_query_time = min(q['duration_ms'] for q in recent_queries)
                
                # Group by query type
                query_types = {}
                for query in recent_queries:
                    query_type = query['type']
                    if query_type not in query_types:
                        query_types[query_type] = []
                    query_types[query_type].append(query['duration_ms'])
                
                type_stats = {}
                for query_type, times in query_types.items():
                    type_stats[query_type] = {
                        'count': len(times),
                        'avg_ms': sum(times) / len(times),
                        'max_ms': max(times),
                        'min_ms': min(times)
                    }
                
                return JSONResponse(content={
                    "total_queries": len(recent_queries),
                    "avg_query_time_ms": avg_query_time,
                    "max_query_time_ms": max_query_time,
                    "min_query_time_ms": min_query_time,
                    "query_type_stats": type_stats,
                    "connection_pool_stats": metrics_collector._db_connection_pool_stats,
                    "generated_at": datetime.utcnow().isoformat()
                })
            else:
                return JSONResponse(content={
                    "total_queries": 0,
                    "message": "No database queries recorded",
                    "generated_at": datetime.utcnow().isoformat()
                })
        
    except Exception as e:
        log_error(e, context={"endpoint": "database_performance"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database performance metrics"
        )


@router.get("/websocket")
async def get_websocket_metrics():
    """
    Get WebSocket connection and message metrics.
    
    Returns:
        WebSocket performance metrics
    """
    try:
        metrics_collector = get_metrics_collector()
        
        with metrics_collector._lock:
            websocket_stats = {
                "active_connections": metrics_collector._websocket_connections,
                "total_messages_sent": metrics_collector._websocket_messages_sent,
                "total_messages_received": metrics_collector._websocket_messages_received,
                "total_errors": metrics_collector._websocket_errors,
                "error_rate": (
                    metrics_collector._websocket_errors / 
                    max(metrics_collector._websocket_messages_sent + metrics_collector._websocket_messages_received, 1)
                ),
                "generated_at": datetime.utcnow().isoformat()
            }
        
        return JSONResponse(content=websocket_stats)
        
    except Exception as e:
        log_error(e, context={"endpoint": "websocket_metrics"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve WebSocket metrics"
        )


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all caches to free memory.
    
    Returns:
        Cache clear operation result
    """
    try:
        cache = get_cache()
        
        # Get stats before clearing
        stats_before = cache.get_cache_stats()
        
        # Clear all caches
        cache.game_data_cache.clear()
        cache.momentum_cache.clear()
        cache.api_response_cache.clear()
        cache.ml_prediction_cache.clear()
        
        # Get stats after clearing
        stats_after = cache.get_cache_stats()
        
        return JSONResponse(content={
            "status": "success",
            "message": "All caches cleared successfully",
            "memory_freed_mb": stats_before['total_memory_mb'] - stats_after['total_memory_mb'],
            "cleared_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        log_error(e, context={"endpoint": "clear_cache"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.post("/cache/cleanup")
async def cleanup_expired_cache():
    """
    Clean up expired cache entries.
    
    Returns:
        Cache cleanup operation result
    """
    try:
        cache = get_cache()
        
        # Clean up expired entries
        cleaned_count = cache.cleanup_all()
        
        # Get updated stats
        stats = cache.get_cache_stats()
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Cleaned up {cleaned_count} expired cache entries",
            "entries_cleaned": cleaned_count,
            "current_memory_usage_mb": stats['total_memory_mb'],
            "cleaned_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        log_error(e, context={"endpoint": "cleanup_cache"}, severity=ErrorSeverity.MEDIUM)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup cache"
        )


@router.get("/health")
async def performance_health_check():
    """
    Comprehensive performance health check.
    
    Returns:
        Overall system performance health status
    """
    try:
        metrics_collector = get_metrics_collector()
        cache = get_cache()
        
        # Get current metrics
        alerts = metrics_collector.get_performance_alerts()
        cache_stats = cache.get_cache_stats()
        
        # Determine health status
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        warning_alerts = [a for a in alerts if a.get('severity') == 'warning']
        
        if critical_alerts:
            health_status = "critical"
        elif warning_alerts:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        # Calculate overall performance score (0-100)
        performance_score = 100
        
        # Deduct points for alerts
        performance_score -= len(critical_alerts) * 30
        performance_score -= len(warning_alerts) * 10
        
        # Deduct points for high cache memory usage
        total_cache_utilization = sum(
            stats.get('memory_utilization', 0) 
            for stats in cache_stats.values() 
            if isinstance(stats, dict)
        ) / 4  # Average across 4 caches
        
        if total_cache_utilization > 0.9:
            performance_score -= 20
        elif total_cache_utilization > 0.7:
            performance_score -= 10
        
        performance_score = max(0, performance_score)
        
        return JSONResponse(content={
            "health_status": health_status,
            "performance_score": performance_score,
            "alerts": {
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
                "total": len(alerts)
            },
            "cache_health": {
                "total_memory_mb": cache_stats.get('total_memory_mb', 0),
                "average_utilization": total_cache_utilization
            },
            "recommendations": _get_performance_recommendations(alerts, cache_stats),
            "checked_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        log_error(e, context={"endpoint": "performance_health"}, severity=ErrorSeverity.HIGH)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "health_status": "unknown",
                "error": "Health check failed",
                "checked_at": datetime.utcnow().isoformat()
            }
        )


def _get_performance_recommendations(alerts: list, cache_stats: dict) -> list:
    """Generate performance improvement recommendations."""
    recommendations = []
    
    # Check for slow requests
    slow_request_alerts = [a for a in alerts if a.get('type') == 'slow_requests']
    if slow_request_alerts:
        recommendations.append({
            "type": "optimization",
            "priority": "high",
            "message": "Consider optimizing slow API endpoints or adding caching"
        })
    
    # Check for high memory usage
    if cache_stats.get('total_memory_mb', 0) > 80:
        recommendations.append({
            "type": "memory",
            "priority": "medium",
            "message": "Cache memory usage is high. Consider clearing expired entries or reducing cache sizes"
        })
    
    # Check for database performance
    db_alerts = [a for a in alerts if a.get('type') == 'slow_database']
    if db_alerts:
        recommendations.append({
            "type": "database",
            "priority": "high",
            "message": "Database queries are slow. Consider adding indexes or optimizing queries"
        })
    
    # Check for WebSocket errors
    ws_alerts = [a for a in alerts if a.get('type') == 'websocket_errors']
    if ws_alerts:
        recommendations.append({
            "type": "websocket",
            "priority": "medium",
            "message": "High WebSocket error rate. Check connection stability and error handling"
        })
    
    return recommendations