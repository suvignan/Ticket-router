import yaml
import hashlib

def read_yaml(path) -> dict:
    """Load a YAML file. safe_load only: yaml.load can execute arbitrary code."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path,chunk=1024*1024)->str:
    """Return the SHA256 hash of a file"""
    h = hashlib.sha256()
    with open(path,'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()