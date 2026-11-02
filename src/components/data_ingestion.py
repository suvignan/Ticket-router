import warnings
import pandas as pd

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import TicketRouterException
from src.logger import logger
from src.utils import sha256_file


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            cfg = self.config
            # 1. Fingerprint
            raw_sha256 = sha256_file(cfg.raw_csv)
            logger.info(f"Raw Data Hash: {raw_sha256[:12]}")

            # 2. Read once (read_csv must stay INSIDE the with, or its warnings aren't caught)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                df = pd.read_csv(cfg.raw_csv, usecols=list(cfg.columns), engine="python", on_bad_lines="warn")
            n_rows_read = len(df)             # saved now: df shrinks in steps 3 and 5
            n_bad_lines = sum(str(w.message).startswith("Skipping line") for w in caught)
            logger.info(f"Rows read: {n_rows_read:,} | bad lines: {n_bad_lines}")
            if n_bad_lines > cfg.max_bad_lines:
                raise ValueError(f"{n_bad_lines} bad lines in {cfg.raw_csv} > max {cfg.max_bad_lines}")
            # 3. Junk filter: checks the RAW narrative, BEFORE clean_text
            #    (after cleaning, "XXXX" becomes "redactedinfo" and rule B would stop matching)

            text = df[cfg.text_column].str.strip()
            is_nan = text.isna()
            is_redacted_only = text.str.replace(cfg.redaction_only_chars,"",regex=True).str.len().eq(0)
            is_attachment = (text.str.len().le(cfg.attachment_max_len) & text.str.contains(cfg.attachment_phrase,regex=False,na=False,case=False))
            junk = is_nan | is_redacted_only | is_attachment
            n_junk_removed = int(junk.sum())
            df = df[~junk].copy()
            logger.info(f"Junk Removed: {n_junk_removed:,} | rows left: {len(df):,}")
            #4.Parse Dates 
            df[cfg.date_column] = pd.to_datetime(df[cfg.date_column],format=cfg.date_format,errors="coerce")
            n_bad_dates = int(df[cfg.date_column].isna().sum())
            if n_bad_dates > 0:
                raise ValueError(f"{n_bad_dates} rows have bad dates in {cfg.date_column}")
            logger.info(f"Dates Parsing: {df[cfg.date_column].min().date()}->{df[cfg.date_column].max().date()}")
            # 5. Cutoff: keep rows strictly BEFORE the cutoff (half-open, like the split)
            cutoff = pd.Timestamp(cfg.cutoff_date)
            df = df[df[cfg.date_column] < cutoff].copy()
            n_after_cutoff = len(df)
            logger.info(f"After cutoff (< {cfg.cutoff_date}): {n_after_cutoff:,} rows")
            # 6. Save (Sub-product kept: LabelBuilder needs it, then drops it)
            cfg.output_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cfg.output_parquet, index=False, engine="pyarrow")
            logger.info(f"Saved {len(df):,} rows → {cfg.output_parquet}")

            return DataIngestionArtifact(
                output_parquet=cfg.output_parquet,
                n_rows_read=n_rows_read,
                n_bad_lines=n_bad_lines,
                n_junk_removed=n_junk_removed,
                n_after_cutoff=n_after_cutoff,
                raw_sha256=raw_sha256,
            )

        except Exception as e:
            raise TicketRouterException(e) from e
