#!/usr/bin/env python3
"""
Launch script for MomentumML application.

This script starts both the backend API server and the frontend development server.
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AppLauncher:
    """Launcher for the MomentumML application."""
    
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.root_dir = Path(__file__).parent
        
    def check_dependencies(self):
        """Check if required dependencies are available."""
        logger.info("Checking dependencies...")
        
        # Check Python dependencies
        try:
            import fastapi
            import uvicorn
            import numpy
            import pandas
            import sklearn
            logger.info("✅ Python dependencies available")
        except ImportError as e:
            logger.error(f"❌ Missing Python dependency: {e}")
            logger.info("Install with: pip install -r requirements.txt")
            return False
        
        # Check if production model exists
        model_path = self.root_dir / "models" / "advanced" / "advanced_nba_momentum_random_forest_20251023_172504.pkl"
        if model_path.exists():
            logger.info("✅ Production model found")
        else:
            logger.warning("⚠️  Production model not found - predictions may not work")
        
        # Check Node.js and npm
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Node.js available: {result.stdout.strip()}")
            else:
                logger.error("❌ Node.js not found")
                return False
        except FileNotFoundError:
            logger.error("❌ Node.js not found")
            return False
        
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ npm available: {result.stdout.strip()}")
            else:
                logger.error("❌ npm not found")
                return False
        except FileNotFoundError:
            logger.error("❌ npm not found")
            return False
        
        # Check if frontend dependencies are installed
        frontend_dir = self.root_dir / "frontend"
        node_modules = frontend_dir / "node_modules"
        
        if node_modules.exists():
            logger.info("✅ Frontend dependencies installed")
        else:
            logger.info("📦 Installing frontend dependencies...")
            try:
                subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
                logger.info("✅ Frontend dependencies installed")
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to install frontend dependencies")
                return False
        
        return True
    
    def start_backend(self):
        """Start the backend API server."""
        logger.info("🚀 Starting backend server...")
        
        try:
            # Start backend with uvicorn
            self.backend_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn',
                'backend.main:app',
                '--host', '0.0.0.0',
                '--port', '8000',
                '--reload'
            ], cwd=self.root_dir)
            
            logger.info("✅ Backend server started on http://localhost:8000")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend development server."""
        logger.info("🚀 Starting frontend server...")
        
        frontend_dir = self.root_dir / "frontend"
        
        try:
            # Start frontend with npm
            self.frontend_process = subprocess.Popen([
                'npm', 'start'
            ], cwd=frontend_dir)
            
            logger.info("✅ Frontend server started on http://localhost:3000")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready."""
        import requests
        
        logger.info("⏳ Waiting for backend to be ready...")
        
        for i in range(timeout):
            try:
                response = requests.get('http://localhost:8000/health', timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Backend is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        logger.warning("⚠️  Backend may not be fully ready yet")
        return False
    
    def launch(self):
        """Launch the complete application."""
        logger.info("🎯 Launching MomentumML Application")
        logger.info("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed")
            return False
        
        # Start backend
        if not self.start_backend():
            logger.error("❌ Failed to start backend")
            return False
        
        # Wait for backend to be ready
        self.wait_for_backend()
        
        # Start frontend
        if not self.start_frontend():
            logger.error("❌ Failed to start frontend")
            self.stop()
            return False
        
        logger.info("=" * 50)
        logger.info("🎉 MomentumML Application Launched Successfully!")
        logger.info("=" * 50)
        logger.info("📊 Frontend: http://localhost:3000")
        logger.info("🔧 Backend API: http://localhost:8000")
        logger.info("📋 API Docs: http://localhost:8000/docs")
        logger.info("❤️  Health Check: http://localhost:8000/health")
        logger.info("=" * 50)
        logger.info("Press Ctrl+C to stop the application")
        
        return True
    
    def stop(self):
        """Stop all processes."""
        logger.info("🛑 Stopping application...")
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                logger.info("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                logger.info("🔪 Backend force killed")
            except Exception as e:
                logger.error(f"Error stopping backend: {e}")
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                logger.info("✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                logger.info("🔪 Frontend force killed")
            except Exception as e:
                logger.error(f"Error stopping frontend: {e}")
    
    def run(self):
        """Run the application with proper signal handling."""
        def signal_handler(signum, frame):
            logger.info("\n🛑 Received shutdown signal")
            self.stop()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            if self.launch():
                # Keep the main process alive
                while True:
                    time.sleep(1)
                    
                    # Check if processes are still running
                    if self.backend_process and self.backend_process.poll() is not None:
                        logger.error("❌ Backend process died")
                        break
                    
                    if self.frontend_process and self.frontend_process.poll() is not None:
                        logger.error("❌ Frontend process died")
                        break
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.stop()


def main():
    """Main entry point."""
    launcher = AppLauncher()
    launcher.run()


if __name__ == "__main__":
    main()