from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src import constants
from src.utils import read_yaml


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_csv: Path
    columns: tuple[str, ...]          # raw columns + label-only columns (tuple: lists could be mutated)
    date_column: str
    date_format: str
    max_bad_lines: int
    cutoff_date: date                 # rows on or after this are dropped
    redaction_only_chars: str
    attachment_phrase: str
    attachment_max_len: int
    output_parquet: Path
    text_column:str


def get_ingestion_config(schema_file=constants.SCHEMA_FILE, params_file=constants.PARAMS_FILE) -> DataIngestionConfig:
    schema, params = read_yaml(schema_file), read_yaml(params_file)
    junk = params["junk_rules"]
    return DataIngestionConfig(
        raw_csv=constants.RAW_CSV,
        columns=tuple(schema["raw_columns"]) + tuple(schema["label_only_columns"]),
        date_column=schema["date_column"],
        date_format=schema["date_format"],
        text_column=schema["text_column"],
        max_bad_lines=params["ingestion"]["max_bad_lines"],
        cutoff_date=date.fromisoformat(params["ingestion"]["cutoff_date"]),   # fails early on a bad date
        redaction_only_chars=junk["redaction_only_chars"],
        attachment_phrase=junk["attachment_phrase"],
        attachment_max_len=junk["attachment_max_len"],
        output_parquet=constants.INGESTED_PARQUET,
    )




@dataclass(frozen=True)
class DataValidationConfig:
    input_parquet: Path
    report_path: Path
    columns: tuple[str, ...]            # exact expected columns (raw + label-only)
    id_column: str
    date_column: str
    text_column: str
    product_column: str
    subproduct_column: str
    allowed_products: tuple[str, ...]   # the 14
    cutoff_date: date


def get_validation_config(input_parquet: Path, schema_file=constants.SCHEMA_FILE, params_file=constants.PARAMS_FILE) -> DataValidationConfig:
    schema, params = read_yaml(schema_file), read_yaml(params_file)
    return DataValidationConfig(
        input_parquet=input_parquet,
        report_path=constants.VALIDATION_REPORT,
        columns=tuple(schema["raw_columns"]) + tuple(schema["label_only_columns"]),
        id_column=schema["id_column"],
        date_column=schema["date_column"],
        text_column=schema["text_column"],
        product_column=schema["product_column"],
        subproduct_column=schema["label_only_columns"][0],
        allowed_products=tuple(schema["raw_product_values"]),
        cutoff_date=date.fromisoformat(params["ingestion"]["cutoff_date"]),
    )


@dataclass(frozen=True)
class LabelBuilderConfig:
    input_parquet: Path
    output_parquet: Path
    product_column: str
    subproduct_column: str
    label_column: str
    rename_map: tuple[tuple[str, str], ...]     # (old, new) pairs: a dict could be mutated
    old_card_product: str
    credit_label: str
    prepaid_label: str
    credit_subproducts: tuple[str, ...]
    prepaid_subproducts: tuple[str, ...]
    repair_from_label: str
    repair_sub_product: str
    repair_to_label: str
    labels: tuple[str, ...]                      # the 11
    keep_columns: tuple[str, ...]


def get_label_builder_config(input_parquet: Path, schema_file=constants.SCHEMA_FILE, params_file=constants.PARAMS_FILE) -> LabelBuilderConfig:
    schema, params = read_yaml(schema_file), read_yaml(params_file)
    lab = params["labels"]
    card, repair = lab["card_split"], lab["credit_repair_move"]
    return LabelBuilderConfig(
        input_parquet=input_parquet,
        output_parquet=constants.LABELED_PARQUET,
        product_column=schema["product_column"],
        subproduct_column=schema["label_only_columns"][0],
        label_column=schema["label_column"],
        rename_map=tuple(lab["rename_map"].items()),        # dict -> (old, new) pairs
        old_card_product=card["old_product"],
        credit_label=card["credit_label"],
        prepaid_label=card["prepaid_label"],
        credit_subproducts=tuple(card["credit_subproducts"]),
        prepaid_subproducts=tuple(card["prepaid_subproducts"]),
        repair_from_label=repair["from_label"],
        repair_sub_product=repair["sub_product"],
        repair_to_label=repair["to_label"],
        labels=tuple(schema["labels"]),
        keep_columns=tuple(schema["labeled_columns"]),
    )



@dataclass(frozen=True)
class DataSplitterConfig:
    input_parquet: Path
    output_dir: Path
    id_column: str
    date_column: str
    text_column: str
    label_column: str
    train_end: date              # all *_end are EXCLUSIVE (half-open)
    val_start: date
    val_end: date
    test_start: date
    test_end: date
    seed: int


def get_splitter_config(input_parquet: Path, schema_file=constants.SCHEMA_FILE, params_file=constants.PARAMS_FILE) -> DataSplitterConfig:
    schema, params = read_yaml(schema_file), read_yaml(params_file)
    s = params["split"]
    d = date.fromisoformat                       # short name, used 5 times
    return DataSplitterConfig(
        input_parquet=input_parquet,
        output_dir=constants.SPLIT_DIR,
        id_column=schema["id_column"],
        date_column=schema["date_column"],
        text_column=schema["text_column"],
        label_column=schema["label_column"],
        train_end=d(s["train_end_exclusive"]),
        val_start=d(s["val_start"]),
        val_end=d(s["val_end_exclusive"]),
        test_start=d(s["test_start"]),
        test_end=d(s["test_end_exclusive"]),
        seed=s["seed"],
    )

@dataclass(frozen=True)
class ModelTrainerConfig:
    train_parquet: Path
    output_dir: Path
    text_column: str
    label_column: str
    min_df: int
    ngram_range: tuple[int, int]
    sublinear_tf: bool
    max_iter: int
    C: float
    class_weight: str | None
    labels: tuple[str, ...]          # the 11, in label_order.json order


def get_trainer_config(train_parquet: Path, schema_file=constants.SCHEMA_FILE, params_file=constants.PARAMS_FILE) -> ModelTrainerConfig:
    schema, params = read_yaml(schema_file), read_yaml(params_file)
    tf, lr = params["model_trainer"]["tfidf"], params["model_trainer"]["logreg"]
    return ModelTrainerConfig(
        train_parquet=train_parquet,
        output_dir=constants.MODEL_DIR,
        text_column=schema["text_column"],
        label_column=schema["label_column"],
        min_df=tf["min_df"],
        ngram_range=tuple(tf["ngram_range"]),       # list [1, 1] -> tuple (1, 1): scikit-learn wants a tuple
        sublinear_tf=tf["sublinear_tf"],
        max_iter=lr["max_iter"],
        C=lr["C"],
        class_weight=lr["class_weight"],
        labels=tuple(schema["labels"]),
    )
