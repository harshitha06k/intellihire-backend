import sys
import os
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    host = "0.0.0.0"

    print(f"\n🚀 Starting M2 Sensors API on {host}:{port}")
    print(f"   Docs: http://localhost:{port}/docs")
    print(f"   WebSocket Audio: ws://localhost:{port}/ws/audio")
    print(f"   WebSocket Video: ws://localhost:{port}/ws/video\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
