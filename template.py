"""Create the project skeleton. Never overwrites a file that already exists."""
import os
from pathlib import Path

FILES = [
    "config/schema.yaml",
    "params.yaml",
    "src/__init__.py",
    "src/constants/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/utils/__init__.py",
    "src/logger/__init__.py",
    "src/exception/__init__.py",
    "tests/__init__.py",
]

for f in map(Path, FILES):
    os.makedirs(f.parent, exist_ok=True)
    if f.exists():
        print(f"skip (exists): {f}")
    else:
        f.touch()
        print(f"created:       {f}")
