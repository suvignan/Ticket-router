from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    output_parquet: Path
    n_rows_read: int
    n_bad_lines: int
    n_junk_removed: int
    n_after_cutoff: int
    raw_sha256: str                   # which CSV this run actually used
