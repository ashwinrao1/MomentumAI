#!/usr/bin/env python3
"""
Integration test for WebSocket real-time communication.

This test verifies that the WebSocket endpoints work correctly,
including connection management, heartbeat mechanism, and data streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List

import websockets
from websockets.exceptions import ConnectionClosed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketTestClient:
    """Test client for WebSocket functionality."""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self.received_messages: List[Dict] = []
        self.is_connected = False
    
    async def connect(self):
        """Connect to WebSocket endpoint."""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            logger.info(f"Connected to {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.uri}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket endpoint."""
        if self.websocket and self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from WebSocket")
    
    async def send_message(self, message: Dict):
        """Send a message to the WebSocket."""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.send(json.dumps(message))
                logger.info(f"Sent message: {message}")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
    
    async def receive_messages(self, timeout: float = 5.0):
        """Receive messages from WebSocket with timeout."""
        if not self.websocket or not self.is_connected:
            return []
        
        messages = []
        try:
            # Use asyncio.wait_for to implement timeout
            while True:
                try:
                    message_text = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=timeout
                    )
                    message = json.loads(message_text)
                    messages.append(message)
                    self.received_messages.append(message)
                    logger.info(f"Received message: {message}")
                except asyncio.TimeoutError:
                    break
                except ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    self.is_connected = False
                    break
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
        
        return messages


async def test_websocket_connection():
    """Test basic WebSocket connection and disconnection."""
    logger.info("Testing WebSocket connection...")
    
    client = WebSocketTestClient("ws://localhost:8000/ws/momentum")
    
    # Test connection
    connected = await client.connect()
    assert connected, "Failed to connect to WebSocket"
    
    # Receive welcome message
    messages = await client.receive_messages(timeout=2.0)
    assert len(messages) > 0, "No welcome message received"
    assert messages[0]["type"] == "connected", "Expected connected message"
    
    # Test disconnection
    await client.disconnect()
    
    logger.info("✓ WebSocket connection test passed")


async def test_heartbeat_mechanism():
    """Test heartbeat ping/pong mechanism."""
    logger.info("Testing heartbeat mechanism...")
    
    client = WebSocketTestClient("ws://localhost:8000/ws/momentum")
    
    # Connect
    await client.connect()
    await client.receive_messages(timeout=1.0)  # Clear welcome message
    
    # Send ping
    await client.send_message({"action": "ping"})
    
    # Receive pong
    messages = await client.receive_messages(timeout=2.0)
    pong_received = any(msg.get("type") == "pong" for msg in messages)
    assert pong_received, "No pong response received"
    
    await client.disconnect()
    
    logger.info("✓ Heartbeat mechanism test passed")


async def test_game_subscription():
    """Test game subscription and unsubscription."""
    logger.info("Testing game subscription...")
    
    client = WebSocketTestClient("ws://localhost:8000/ws/momentum")
    
    # Connect
    await client.connect()
    await client.receive_messages(timeout=1.0)  # Clear welcome message
    
    # Subscribe to a game
    test_game_id = "test_game_123"
    await client.send_message({
        "action": "subscribe",
        "game_id": test_game_id
    })
    
    # Receive subscription confirmation
    messages = await client.receive_messages(timeout=2.0)
    subscribed = any(
        msg.get("type") == "subscribed" and msg.get("game_id") == test_game_id 
        for msg in messages
    )
    assert subscribed, "No subscription confirmation received"
    
    # Unsubscribe from game
    await client.send_message({
        "action": "unsubscribe",
        "game_id": test_game_id
    })
    
    # Receive unsubscription confirmation
    messages = await client.receive_messages(timeout=2.0)
    unsubscribed = any(
        msg.get("type") == "unsubscribed" and msg.get("game_id") == test_game_id 
        for msg in messages
    )
    assert unsubscribed, "No unsubscription confirmation received"
    
    await client.disconnect()
    
    logger.info("✓ Game subscription test passed")


async def test_multiple_connections():
    """Test multiple concurrent WebSocket connections."""
    logger.info("Testing multiple connections...")
    
    clients = []
    num_clients = 3
    
    # Create multiple clients
    for i in range(num_clients):
        client = WebSocketTestClient("ws://localhost:8000/ws/momentum")
        await client.connect()
        clients.append(client)
    
    # Clear welcome messages
    for client in clients:
        await client.receive_messages(timeout=1.0)
    
    # Test that all clients can send/receive messages
    for i, client in enumerate(clients):
        await client.send_message({"action": "ping"})
        messages = await client.receive_messages(timeout=2.0)
        pong_received = any(msg.get("type") == "pong" for msg in messages)
        assert pong_received, f"Client {i} did not receive pong"
    
    # Disconnect all clients
    for client in clients:
        await client.disconnect()
    
    logger.info("✓ Multiple connections test passed")


async def test_error_handling():
    """Test error handling for invalid messages."""
    logger.info("Testing error handling...")
    
    client = WebSocketTestClient("ws://localhost:8000/ws/momentum")
    
    # Connect
    await client.connect()
    await client.receive_messages(timeout=1.0)  # Clear welcome message
    
    # Send invalid JSON
    if client.websocket:
        await client.websocket.send("invalid json")
    
    # Receive error message
    messages = await client.receive_messages(timeout=2.0)
    error_received = any(msg.get("type") == "error" for msg in messages)
    assert error_received, "No error message received for invalid JSON"
    
    # Send unknown action
    await client.send_message({"action": "unknown_action"})
    
    # Receive error message
    messages = await client.receive_messages(timeout=2.0)
    error_received = any(
        msg.get("type") == "error" and "Unknown action" in msg.get("message", "")
        for msg in messages
    )
    assert error_received, "No error message received for unknown action"
    
    await client.disconnect()
    
    logger.info("✓ Error handling test passed")


async def run_websocket_tests():
    """Run all WebSocket tests."""
    logger.info("Starting WebSocket integration tests...")
    
    tests = [
        test_websocket_connection,
        test_heartbeat_mechanism,
        test_game_subscription,
        test_multiple_connections,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            failed += 1
    
    logger.info(f"WebSocket tests completed: {passed} passed, {failed} failed")
    
    if failed > 0:
        raise Exception(f"{failed} tests failed")


if __name__ == "__main__":
    print("WebSocket Integration Test")
    print("=" * 50)
    print("This test requires the FastAPI server to be running on localhost:8000")
    print("Start the server with: python backend/main.py")
    print("Then run this test in another terminal")
    print("=" * 50)
    
    try:
        asyncio.run(run_websocket_tests())
        print("\n✅ All WebSocket tests passed!")
    except Exception as e:
        print(f"\n❌ WebSocket tests failed: {e}")
        exit(1)