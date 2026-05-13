import sys
import os
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    host = "0.0.0.0"

    print(f"\n🚀 Starting M2 Sensors API on {host}:{port}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
