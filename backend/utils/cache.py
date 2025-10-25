"""
In-memory caching system for MomentumML performance optimization.

This module provides a high-performance caching layer for frequently accessed data,
including momentum calculations, game data, and API responses.
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from threading import RLock
import weakref

logger = logging.getLogger("momentum_ml.cache")


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """
    Thread-safe LRU cache with TTL support and memory management.
    
    Features:
    - Least Recently Used eviction policy
    - Time-to-live (TTL) expiration
    - Memory usage tracking
    - Thread-safe operations
    - Statistics tracking
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_evictions': 0,
            'expired_evictions': 0,
            'total_memory_bytes': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['misses'] += 1
                self._stats['expired_evictions'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats['hits'] += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put a value in the cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            current_time = time.time()
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Ensure we have space
            self._ensure_capacity(size_bytes)
            
            # Add to cache
            self._cache[key] = entry
            self._stats['total_memory_bytes'] += size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a specific key from the cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._stats['total_memory_bytes'] = 0
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries and return count removed."""
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats['expired_evictions'] += 1
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._stats['total_memory_bytes'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'memory_utilization': self._stats['total_memory_bytes'] / self.max_memory_bytes
            }
    
    def _remove_entry(self, key: str):
        """Remove an entry and update memory tracking."""
        entry = self._cache.pop(key, None)
        if entry:
            self._stats['total_memory_bytes'] -= entry.size_bytes
    
    def _ensure_capacity(self, new_item_size: int):
        """Ensure cache has capacity for new item."""
        # Remove expired items first
        self.cleanup_expired()
        
        # Check size limit
        while len(self._cache) >= self.max_size:
            self._evict_lru()
        
        # Check memory limit
        while (self._stats['total_memory_bytes'] + new_item_size) > self.max_memory_bytes:
            if not self._cache:
                break
            self._evict_lru()
            self._stats['memory_evictions'] += 1
    
    def _evict_lru(self):
        """Evict the least recently used item."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)  # Remove first (oldest)
            self._stats['evictions'] += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the memory size of a value."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value) + 64
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v) 
                    for k, v in value.items()
                ) + 64
            else:
                # Fallback: serialize to JSON and measure
                return len(json.dumps(value, default=str))
        except Exception:
            return 1024  # Default estimate


class MomentumCache:
    """
    Specialized cache for MomentumML data with domain-specific optimizations.
    
    Provides separate cache layers for different data types with appropriate
    TTL values and eviction policies.
    """
    
    def __init__(self):
        # Different cache layers with appropriate configurations
        self.game_data_cache = LRUCache(max_size=100, max_memory_mb=20)  # Game info, events
        self.momentum_cache = LRUCache(max_size=500, max_memory_mb=30)   # TMI calculations
        self.api_response_cache = LRUCache(max_size=200, max_memory_mb=10)  # API responses
        self.ml_prediction_cache = LRUCache(max_size=300, max_memory_mb=15)  # ML predictions
        
        # Default TTL values (seconds)
        self.ttl_config = {
            'game_data': 30,      # Game data expires quickly (live updates)
            'momentum': 60,       # Momentum calculations cached longer
            'api_response': 20,   # API responses cached briefly
            'ml_prediction': 120, # ML predictions cached longer (expensive)
            'static_data': 3600   # Static data cached for 1 hour
        }
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def get_game_data(self, game_id: str) -> Optional[Any]:
        """Get cached game data."""
        return self.game_data_cache.get(f"game_data:{game_id}")
    
    def cache_game_data(self, game_id: str, data: Any) -> bool:
        """Cache game data with appropriate TTL."""
        return self.game_data_cache.put(
            f"game_data:{game_id}", 
            data, 
            ttl=self.ttl_config['game_data']
        )
    
    def get_momentum(self, game_id: str, team_tricode: str) -> Optional[Any]:
        """Get cached momentum data."""
        return self.momentum_cache.get(f"momentum:{game_id}:{team_tricode}")
    
    def cache_momentum(self, game_id: str, team_tricode: str, tmi_data: Any) -> bool:
        """Cache momentum data with appropriate TTL."""
        return self.momentum_cache.put(
            f"momentum:{game_id}:{team_tricode}",
            tmi_data,
            ttl=self.ttl_config['momentum']
        )
    
    def get_api_response(self, endpoint: str, params: str = "") -> Optional[Any]:
        """Get cached API response."""
        cache_key = f"api:{endpoint}:{hash(params)}"
        return self.api_response_cache.get(cache_key)
    
    def cache_api_response(self, endpoint: str, params: str, response: Any) -> bool:
        """Cache API response with appropriate TTL."""
        cache_key = f"api:{endpoint}:{hash(params)}"
        return self.api_response_cache.put(
            cache_key,
            response,
            ttl=self.ttl_config['api_response']
        )
    
    def get_ml_prediction(self, feature_hash: str) -> Optional[Any]:
        """Get cached ML prediction."""
        return self.ml_prediction_cache.get(f"ml_pred:{feature_hash}")
    
    def cache_ml_prediction(self, feature_hash: str, prediction: Any) -> bool:
        """Cache ML prediction with appropriate TTL."""
        return self.ml_prediction_cache.put(
            f"ml_pred:{feature_hash}",
            prediction,
            ttl=self.ttl_config['ml_prediction']
        )
    
    def invalidate_game(self, game_id: str):
        """Invalidate all cached data for a specific game."""
        # This is a simplified approach - in production you might want
        # more sophisticated cache invalidation
        patterns = [
            f"game_data:{game_id}",
            f"momentum:{game_id}:",
        ]
        
        for cache in [self.game_data_cache, self.momentum_cache]:
            keys_to_delete = []
            with cache._lock:
                for key in cache._cache.keys():
                    for pattern in patterns:
                        if key.startswith(pattern):
                            keys_to_delete.append(key)
            
            for key in keys_to_delete:
                cache.delete(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'game_data_cache': self.game_data_cache.get_stats(),
            'momentum_cache': self.momentum_cache.get_stats(),
            'api_response_cache': self.api_response_cache.get_stats(),
            'ml_prediction_cache': self.ml_prediction_cache.get_stats(),
            'total_memory_mb': sum([
                cache.get_stats()['memory_usage_mb'] 
                for cache in [
                    self.game_data_cache, 
                    self.momentum_cache, 
                    self.api_response_cache, 
                    self.ml_prediction_cache
                ]
            ])
        }
    
    def cleanup_all(self):
        """Clean up expired entries in all caches."""
        total_cleaned = 0
        for cache in [self.game_data_cache, self.momentum_cache, 
                     self.api_response_cache, self.ml_prediction_cache]:
            total_cleaned += cache.cleanup_expired()
        
        logger.info(f"Cleaned up {total_cleaned} expired cache entries")
        return total_cleaned
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Clean up every 5 minutes
                    self.cleanup_all()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            logger.info("No event loop for cache cleanup task")


# Global cache instance
_global_cache: Optional[MomentumCache] = None


def get_cache() -> MomentumCache:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MomentumCache()
    return _global_cache


def clear_cache():
    """Clear all caches."""
    cache = get_cache()
    cache.game_data_cache.clear()
    cache.momentum_cache.clear()
    cache.api_response_cache.clear()
    cache.ml_prediction_cache.clear()


# Decorator for caching function results
def cached(cache_type: str = 'api_response', ttl: Optional[float] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_type: Type of cache to use ('api_response', 'momentum', etc.)
        ttl: Time to live in seconds (uses default if None)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            if cache_type == 'momentum':
                result = cache.momentum_cache.get(cache_key)
            elif cache_type == 'ml_prediction':
                result = cache.ml_prediction_cache.get(cache_key)
            else:
                result = cache.api_response_cache.get(cache_key)
            
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Cache the result
            if cache_type == 'momentum':
                cache.momentum_cache.put(cache_key, result, ttl)
            elif cache_type == 'ml_prediction':
                cache.ml_prediction_cache.put(cache_key, result, ttl)
            else:
                cache.api_response_cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator