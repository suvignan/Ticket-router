import dataclasses
import hashlib
import json
import typing
from pathlib import Path

import yaml


def read_yaml(path) -> dict:
    """Load a YAML file. safe_load only: yaml.load can execute arbitrary code."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path, chunk=1024 * 1024) -> str:
    """Return the SHA256 hash of a file"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def save_artifact(artifact, path) -> None:
    """Artifact dataclass -> JSON (DVC stages talk only through files). default=str turns Path into str."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(artifact), indent=2, default=str), encoding="utf-8")


def load_artifact(cls, path):
    """JSON -> artifact dataclass. JSON has no Path or tuple, so convert those fields back."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for f in dataclasses.fields(cls):
        if f.type is Path:
            data[f.name] = Path(data[f.name])
        elif typing.get_origin(f.type) is tuple:
            data[f.name] = tuple(data[f.name])
    return cls(**data)
