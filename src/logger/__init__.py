import logging
import os
from datetime import datetime
from pathlib import Path

# Anchored to the project root, so running from notebooks/ doesn't create notebooks/logs/
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now():%Y_%m_%d_%H_%M_%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("ticket_router")
