"""
Performance monitoring and metrics collection for MomentumML.

This module provides comprehensive performance monitoring including:
- Request timing and throughput metrics
- Database query performance
- WebSocket connection metrics
- Memory and CPU usage tracking
- Cache hit rates and efficiency
"""

import asyncio
import logging
import psutil
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from threading import Lock
import weakref

logger = logging.getLogger("momentum_ml.performance")


@dataclass
class PerformanceMetric:
    """Represents a single performance metric."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class TimingMetric:
    """Represents timing information for an operation."""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    
    Features:
    - Request timing and throughput
    - Database query performance
    - WebSocket metrics
    - System resource usage
    - Cache performance
    """
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self._lock = Lock()
        
        # Metric storage
        self._timing_metrics: deque = deque(maxlen=max_history_size)
        self._counter_metrics: Dict[str, int] = defaultdict(int)
        self._gauge_metrics: Dict[str, float] = {}
        self._histogram_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Request tracking
        self._active_requests: Dict[str, float] = {}
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._request_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Database metrics
        self._db_query_times: deque = deque(maxlen=1000)
        self._db_connection_pool_stats: Dict[str, Any] = {}
        
        # WebSocket metrics
        self._websocket_connections: int = 0
        self._websocket_messages_sent: int = 0
        self._websocket_messages_received: int = 0
        self._websocket_errors: int = 0
        
        # System metrics
        self._system_metrics_history: deque = deque(maxlen=100)
        
        # Start background collection
        self._collection_task = None
        self._start_background_collection()
    
    def record_timing(self, operation: str, duration_ms: float, 
                     success: bool = True, error_message: Optional[str] = None,
                     tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        with self._lock:
            metric = TimingMetric(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                success=success,
                error_message=error_message,
                tags=tags or {}
            )
            self._timing_metrics.append(metric)
            
            # Update histogram
            self._histogram_metrics[f"{operation}_duration"].append(duration_ms)
            
            # Update counters
            self._counter_metrics[f"{operation}_total"] += 1
            if success:
                self._counter_metrics[f"{operation}_success"] += 1
            else:
                self._counter_metrics[f"{operation}_error"] += 1
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = name
            if tags:
                tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
                key = f"{name}[{tag_str}]"
            self._counter_metrics[key] += value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            key = name
            if tags:
                tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
                key = f"{name}[{tag_str}]"
            self._gauge_metrics[key] = value
    
    def record_histogram(self, name: str, value: float):
        """Record a value in a histogram."""
        with self._lock:
            self._histogram_metrics[name].append(value)
    
    def start_request(self, request_id: str, endpoint: str):
        """Start tracking a request."""
        with self._lock:
            self._active_requests[request_id] = time.time()
            self._request_counts[endpoint] += 1
    
    def end_request(self, request_id: str, endpoint: str, success: bool = True):
        """End tracking a request."""
        with self._lock:
            start_time = self._active_requests.pop(request_id, None)
            if start_time:
                duration_ms = (time.time() - start_time) * 1000
                self._request_durations[endpoint].append(duration_ms)
                self.record_timing(f"request_{endpoint}", duration_ms, success)
    
    def record_db_query(self, query_type: str, duration_ms: float, success: bool = True):
        """Record database query performance."""
        with self._lock:
            self._db_query_times.append({
                'type': query_type,
                'duration_ms': duration_ms,
                'timestamp': datetime.utcnow(),
                'success': success
            })
            self.record_timing(f"db_query_{query_type}", duration_ms, success)
    
    def update_db_pool_stats(self, stats: Dict[str, Any]):
        """Update database connection pool statistics."""
        with self._lock:
            self._db_connection_pool_stats = stats
            
            # Set gauge metrics for pool stats
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    self.set_gauge(f"db_pool_{key}", float(value))
    
    def record_websocket_connection(self, connected: bool):
        """Record WebSocket connection change."""
        with self._lock:
            if connected:
                self._websocket_connections += 1
                self.increment_counter("websocket_connections_total")
            else:
                self._websocket_connections = max(0, self._websocket_connections - 1)
                self.increment_counter("websocket_disconnections_total")
            
            self.set_gauge("websocket_active_connections", self._websocket_connections)
    
    def record_websocket_message(self, sent: bool = True, error: bool = False):
        """Record WebSocket message activity."""
        with self._lock:
            if error:
                self._websocket_errors += 1
                self.increment_counter("websocket_errors_total")
            elif sent:
                self._websocket_messages_sent += 1
                self.increment_counter("websocket_messages_sent_total")
            else:
                self._websocket_messages_received += 1
                self.increment_counter("websocket_messages_received_total")
    
    def collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk usage (for database)
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            
            system_metrics = {
                'timestamp': datetime.utcnow(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_mb': memory_used_mb,
                'disk_percent': disk_percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }
            
            with self._lock:
                self._system_metrics_history.append(system_metrics)
                
                # Update gauge metrics
                self.set_gauge("system_cpu_percent", cpu_percent)
                self.set_gauge("system_memory_percent", memory_percent)
                self.set_gauge("system_memory_used_mb", memory_used_mb)
                self.set_gauge("system_disk_percent", disk_percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_metrics_summary(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Get a summary of metrics for the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            # Filter recent timing metrics
            recent_timings = [
                m for m in self._timing_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            # Calculate timing statistics
            timing_stats = {}
            operations = set(m.operation for m in recent_timings)
            
            for operation in operations:
                op_timings = [m.duration_ms for m in recent_timings if m.operation == operation]
                if op_timings:
                    timing_stats[operation] = {
                        'count': len(op_timings),
                        'avg_ms': sum(op_timings) / len(op_timings),
                        'min_ms': min(op_timings),
                        'max_ms': max(op_timings),
                        'p95_ms': self._percentile(op_timings, 95),
                        'p99_ms': self._percentile(op_timings, 99)
                    }
            
            # Request statistics
            request_stats = {}
            for endpoint, durations in self._request_durations.items():
                if durations:
                    recent_durations = list(durations)[-50:]  # Last 50 requests
                    request_stats[endpoint] = {
                        'count': self._request_counts[endpoint],
                        'avg_duration_ms': sum(recent_durations) / len(recent_durations),
                        'min_duration_ms': min(recent_durations),
                        'max_duration_ms': max(recent_durations)
                    }
            
            # Database statistics
            recent_db_queries = [
                q for q in self._db_query_times 
                if q['timestamp'] >= cutoff_time
            ]
            
            db_stats = {
                'total_queries': len(recent_db_queries),
                'avg_query_time_ms': (
                    sum(q['duration_ms'] for q in recent_db_queries) / len(recent_db_queries)
                    if recent_db_queries else 0
                ),
                'pool_stats': self._db_connection_pool_stats.copy()
            }
            
            # WebSocket statistics
            websocket_stats = {
                'active_connections': self._websocket_connections,
                'messages_sent': self._websocket_messages_sent,
                'messages_received': self._websocket_messages_received,
                'errors': self._websocket_errors
            }
            
            # System statistics (latest)
            system_stats = {}
            if self._system_metrics_history:
                latest_system = self._system_metrics_history[-1]
                system_stats = {
                    k: v for k, v in latest_system.items() 
                    if k != 'timestamp'
                }
            
            return {
                'time_window_minutes': time_window_minutes,
                'generated_at': datetime.utcnow().isoformat(),
                'timing_stats': timing_stats,
                'request_stats': request_stats,
                'database_stats': db_stats,
                'websocket_stats': websocket_stats,
                'system_stats': system_stats,
                'counter_metrics': dict(self._counter_metrics),
                'gauge_metrics': dict(self._gauge_metrics),
                'active_requests': len(self._active_requests)
            }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds."""
        alerts = []
        
        with self._lock:
            # Check system metrics
            if self._system_metrics_history:
                latest = self._system_metrics_history[-1]
                
                if latest['cpu_percent'] > 80:
                    alerts.append({
                        'type': 'high_cpu',
                        'severity': 'warning',
                        'message': f"High CPU usage: {latest['cpu_percent']:.1f}%",
                        'value': latest['cpu_percent'],
                        'threshold': 80
                    })
                
                if latest['memory_percent'] > 85:
                    alerts.append({
                        'type': 'high_memory',
                        'severity': 'warning',
                        'message': f"High memory usage: {latest['memory_percent']:.1f}%",
                        'value': latest['memory_percent'],
                        'threshold': 85
                    })
            
            # Check request performance
            for endpoint, durations in self._request_durations.items():
                if durations:
                    recent_avg = sum(list(durations)[-10:]) / min(len(durations), 10)
                    if recent_avg > 2000:  # 2 seconds
                        alerts.append({
                            'type': 'slow_requests',
                            'severity': 'warning',
                            'message': f"Slow requests on {endpoint}: {recent_avg:.0f}ms avg",
                            'value': recent_avg,
                            'threshold': 2000,
                            'endpoint': endpoint
                        })
            
            # Check database performance
            if self._db_query_times:
                recent_queries = list(self._db_query_times)[-20:]
                avg_query_time = sum(q['duration_ms'] for q in recent_queries) / len(recent_queries)
                
                if avg_query_time > 1000:  # 1 second
                    alerts.append({
                        'type': 'slow_database',
                        'severity': 'warning',
                        'message': f"Slow database queries: {avg_query_time:.0f}ms avg",
                        'value': avg_query_time,
                        'threshold': 1000
                    })
            
            # Check WebSocket errors
            if self._websocket_errors > 10:
                alerts.append({
                    'type': 'websocket_errors',
                    'severity': 'warning',
                    'message': f"High WebSocket error count: {self._websocket_errors}",
                    'value': self._websocket_errors,
                    'threshold': 10
                })
        
        return alerts
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _start_background_collection(self):
        """Start background system metrics collection."""
        async def collection_loop():
            while True:
                try:
                    self.collect_system_metrics()
                    await asyncio.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    logger.error(f"System metrics collection error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        try:
            loop = asyncio.get_event_loop()
            self._collection_task = loop.create_task(collection_loop())
        except RuntimeError:
            # No event loop running
            logger.info("No event loop for metrics collection task")


# Global metrics collector
_global_metrics: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


@contextmanager
def time_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager to time an operation."""
    start_time = time.time()
    success = True
    error_message = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        get_metrics_collector().record_timing(
            operation_name, 
            duration_ms, 
            success, 
            error_message, 
            tags
        )


def timed(operation_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with time_operation(op_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def async_timed(operation_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Decorator to time async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with time_operation(op_name, tags):
                return await func(*args, **kwargs)
        return wrapper
    return decorator