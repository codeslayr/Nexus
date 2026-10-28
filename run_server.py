"""
Application runner script.
Sets WindowsSelectorEventLoopPolicy before Uvicorn boots to guarantee
100% compatibility with async psycopg3 connection pooling on Windows.
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, log_level="info")
