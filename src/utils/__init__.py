import dataclasses
import hashlib
import json
import re
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



def verify_checksums(paths, manifest, root) -> dict:
    """sha256 of every path (relative to root) == models/manifest.json, else raise. ONE check, used by the API at
    startup (refuses to start) and by CD before `docker build` (refuses to build). Returns {path: sha256}."""
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))["checksums"]
    checksums = {}
    for path in paths:
        actual = sha256_file(Path(root) / path)
        if actual != expected[path]:
            raise RuntimeError(f"{path}: sha256 {actual[:12]} != manifest {expected[path][:12]}. Refusing to start.")
        checksums[path] = actual
    return checksums

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


def slugify(name: str) -> str:
    """MLflow metric names allow only letters, digits, _ - . space /. Labels have commas: lowercase, rest -> _."""
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


def flatten(d: dict, prefix: str = "") -> dict:
    """{'tfidf': {'min_df': 5}} -> {'tfidf.min_df': 5}. MLflow params must be flat; lists become strings."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v) if isinstance(v, (list, tuple)) else v
    return out
