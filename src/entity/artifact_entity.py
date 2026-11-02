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


@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    report_path: Path
    validated_parquet: Path        # same as the input path: validation never copies


@dataclass(frozen=True)
class LabelBuilderArtifact:
    output_parquet: Path
    n_rows: int
    n_renamed: int
    n_card_to_credit: int
    n_card_to_prepaid: int
    n_credit_repair_moved: int
    label_counts: dict          # label -> count (a record of the result, read-only by convention)
