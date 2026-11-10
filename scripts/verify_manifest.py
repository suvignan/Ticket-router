import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import read_yaml, verify_checksums

files = {p for model in read_yaml(ROOT / "params.yaml")["serving"]["files"].values() for p in model.values()}
for path, sha in sorted(verify_checksums(sorted(files), ROOT / "models" / "manifest.json", ROOT).items()):
    print(f"OK {sha[:12]}  {path}")
