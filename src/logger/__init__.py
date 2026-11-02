import logging
import os
from datetime import datetime
from pathlib import Path

# Anchored to the project root, so running from notebooks/ doesn't create notebooks/logs/
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now():%Y_%m_%d_%H_%M_%S}.log"

# Handlers go on OUR logger, not via basicConfig: basicConfig silently does nothing
# if anything (pytest, Jupyter, a library) configured the root logger first.
logger = logging.getLogger("ticket_router")
logger.setLevel(logging.INFO)
logger.propagate = False              # don't print twice through the root logger
if not logger.handlers:               # re-import (e.g. notebook re-run) must not add duplicates
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    for handler in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
