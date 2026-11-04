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

@dataclass(frozen=True)
class DataSplitterArtifact:
    train_parquet: Path
    val_parquet: Path
    test_parquet: Path
    test_dedup_parquet: Path
    n_train: int
    n_val: int
    n_test: int
    n_test_dedup: int
    n_train_removed_dups: int
    n_ties: int
    n_test_seen_in_train: int
    
@dataclass(frozen=True)
class ModelTrainerArtifact:
    tfidf_path: Path
    logreg_path: Path
    label_order_path: Path
    tfidf_sha256: str
    logreg_sha256: str
    label_order_sha256: str
    vocab_size: int
    n_train: int
    train_seconds: float


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    gate_passed: bool
    reasons: tuple[str, ...]                     # empty = passed
    challenger_macro_f1: float
    champion_macro_f1: float | None              # None = no champion
    macro_f1_diff: float | None
    per_class_f1_diff: dict[str, float] | None   # class -> challenger minus champion
    report_path: Path
