import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Get config from environment variables (Docker friendly)
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"Starting API server at http://{host}:{port}")

    # Start Uvicorn server programmatically
    # We point to "src.api.main:app" string to enable reload support if needed
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,  # Set to True for dev, False for prod
        log_level="info",
    )
