import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Get config from environment variables (Docker passes these)
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"🚀 Starting API server at http://{host}:{port}")

    # Start Uvicorn server programmatically
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production (Docker)
        log_level="info",
    )
