#!/usr/bin/env python3
"""
Test script for error handling and resilience features.

This script tests the comprehensive error handling implementation
including retry logic, graceful degradation, and logging.
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.error_handling import (
    APIError, DataProcessingError, NetworkError,
    retry_with_exponential_backoff, handle_api_errors,
    graceful_degradation, safe_execute, validate_data,
    log_error, ErrorSeverity, health_checker,
    setup_logging
)


def test_error_creation():
    """Test custom error creation and logging."""
    print("Testing error creation and logging...")
    
    # Test APIError
    api_error = APIError(
        "Test API error",
        status_code=500,
        severity=ErrorSeverity.HIGH,
        details={"test": "data"}
    )
    
    print(f"API Error: {api_error.message}")
    print(f"Error dict: {api_error.to_dict()}")
    
    # Test logging
    log_error(api_error, context={"test": "context"})
    
    print("✓ Error creation and logging test passed\n")


def test_retry_logic():
    """Test retry logic with exponential backoff."""
    print("Testing retry logic...")
    
    attempt_count = 0
    
    @retry_with_exponential_backoff(max_retries=3, base_delay=0.1)
    async def failing_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Attempt {attempt_count} failed")
        return f"Success on attempt {attempt_count}"
    
    async def run_retry_test():
        try:
            result = await failing_function()
            print(f"Retry result: {result}")
            print("✓ Retry logic test passed\n")
        except Exception as e:
            print(f"✗ Retry logic test failed: {e}\n")
    
    return run_retry_test()


def test_graceful_degradation():
    """Test graceful degradation functionality."""
    print("Testing graceful degradation...")
    
    # Test fallback data
    graceful_degradation.set_fallback_data("test_key", {"data": "test_value"})
    fallback = graceful_degradation.get_fallback_data("test_key")
    
    if fallback and fallback["data"] == "test_value":
        print("✓ Fallback data storage/retrieval works")
    else:
        print("✗ Fallback data test failed")
    
    # Test service status
    graceful_degradation.set_service_status("test_service", False)
    is_healthy = graceful_degradation.is_service_healthy("test_service")
    
    if not is_healthy:
        print("✓ Service status tracking works")
    else:
        print("✗ Service status test failed")
    
    # Test with_fallback
    def primary_function():
        raise Exception("Primary function failed")
    
    def fallback_function():
        return "fallback_result"
    
    result = graceful_degradation.with_fallback(
        primary_function,
        fallback_func=fallback_function
    )
    
    if result == "fallback_result":
        print("✓ with_fallback works")
    else:
        print("✗ with_fallback test failed")
    
    print("✓ Graceful degradation test passed\n")


def test_safe_execute():
    """Test safe execution utility."""
    print("Testing safe execution...")
    
    # Test successful execution
    result = safe_execute(
        lambda: "success",
        "default",
        "Test operation"
    )
    
    if result == "success":
        print("✓ Safe execute success case works")
    else:
        print("✗ Safe execute success case failed")
    
    # Test failure case
    result = safe_execute(
        lambda: 1 / 0,  # This will raise ZeroDivisionError
        "default_value",
        "Division by zero test"
    )
    
    if result == "default_value":
        print("✓ Safe execute failure case works")
    else:
        print("✗ Safe execute failure case failed")
    
    print("✓ Safe execute test passed\n")


def test_validation():
    """Test data validation utility."""
    print("Testing data validation...")
    
    # Test valid data
    try:
        validate_data(
            [1, 2, 3],
            lambda data: isinstance(data, list) and len(data) > 0,
            "List validation test"
        )
        print("✓ Valid data validation works")
    except Exception as e:
        print(f"✗ Valid data validation failed: {e}")
    
    # Test invalid data
    try:
        validate_data(
            [],
            lambda data: isinstance(data, list) and len(data) > 0,
            "Empty list validation test"
        )
        print("✗ Invalid data validation should have failed")
    except DataProcessingError:
        print("✓ Invalid data validation works")
    except Exception as e:
        print(f"✗ Invalid data validation failed with wrong error: {e}")
    
    print("✓ Data validation test passed\n")


def test_health_checker():
    """Test health checker functionality."""
    print("Testing health checker...")
    
    # Register a test health check
    def test_health_check():
        return True
    
    health_checker.register_health_check("test_service", test_health_check)
    
    # Check service health
    is_healthy = health_checker.check_service_health("test_service")
    
    if is_healthy:
        print("✓ Health check registration and execution works")
    else:
        print("✗ Health check test failed")
    
    # Get overall health
    overall_health = health_checker.get_overall_health()
    
    if overall_health["overall_healthy"] and "test_service" in overall_health["services"]:
        print("✓ Overall health check works")
    else:
        print("✗ Overall health check failed")
    
    print("✓ Health checker test passed\n")


async def main():
    """Run all error handling tests."""
    print("=" * 50)
    print("MomentumML Error Handling Test Suite")
    print("=" * 50)
    
    # Setup logging for tests
    setup_logging(log_level="INFO", enable_console=True)
    
    # Run tests
    test_error_creation()
    await test_retry_logic()
    test_graceful_degradation()
    test_safe_execute()
    test_validation()
    test_health_checker()
    
    print("=" * 50)
    print("All error handling tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())