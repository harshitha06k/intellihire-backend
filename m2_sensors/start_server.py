#!/usr/bin/env python3
"""
Start M2 Sensors API Server
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("M2_PORT", "8001"))
    host = os.getenv("M2_HOST", "0.0.0.0")
    
    print(f"\n🚀 Starting M2 Sensors API on {host}:{port}")
    print(f"   Docs: http://localhost:{port}/docs")
    print(f"   WebSocket Audio: ws://localhost:{port}/ws/audio")
    print(f"   WebSocket Video: ws://localhost:{port}/ws/video\n")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
