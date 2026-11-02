import json

import pandas as pd
from pandas.api import types as ptypes

from src.entity.artifact_entity import DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import TicketRouterException
from src.logger import logger


def check_columns(df, expected):
    missing = sorted(set(expected) - set(df.columns))
    extra = sorted(set(df.columns) - set(expected))
    return (not missing and not extra), {"missing": missing, "extra": extra}


def check_dtypes(df, date_column, id_column):
    bad = {}
    for col in df.columns:
        if col == date_column:
            ok = ptypes.is_datetime64_any_dtype(df[col])
        elif col == id_column:
            ok = ptypes.is_integer_dtype(df[col])
        else:
            ok = ptypes.is_string_dtype(df[col])
        if not ok:
            bad[col] = str(df[col].dtype)          # record the dtype it actually has
    return (len(bad) == 0), {"wrong_dtypes": bad}


def check_unique_ids(df, id_column):
    dupes = df[id_column][df[id_column].duplicated(keep=False)]
    return (len(dupes) == 0), {"n_duplicates": int(len(dupes)), "examples": dupes.head(5).tolist()}


def check_products(df, product_column, allowed):
    n_missing = int(df[product_column].isna().sum())
    unexpected = sorted(set(df[product_column].dropna().unique()) - set(allowed))
    passed = len(unexpected) == 0 and n_missing == 0
    return passed, {"unexpected": unexpected, "n_missing": n_missing}



def check_dates(df, date_column, cutoff):
    dates = df[date_column]
    n_after = int((dates >=pd.Timestamp(cutoff)).sum())
    n_with_time = int((dates!=dates.dt.normalize()).sum())
    passed =n_after == 0 and n_with_time == 0
    return passed, {"n_on_or_after_cutoff": n_after, "n_with_time_part": n_with_time,
                    "max_date": str(dates.max().date()) if len(dates) else None}


def check_no_empty_text(df, text_column):
    text = df[text_column]
    n_empty = int((text.isna() | text.str.strip().eq("")).sum())
    return n_empty == 0, {"n_empty": n_empty}


def check_subproduct_present(df, subproduct_column):
    n_missing = int(df[subproduct_column].isna().sum())
    return n_missing == 0, {"n_missing": n_missing}


def check_not_empty(df):
    return len(df) > 0, {"n_rows": int(len(df))}


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            cfg = self.config
            df = pd.read_parquet(cfg.input_parquet)

            checks = {"columns": check_columns(df, cfg.columns)}
            if checks["columns"][0]:          # only if the table has the right shape
                checks.update({
                    "dtypes": check_dtypes(df, cfg.date_column, cfg.id_column),
                    "unique_ids": check_unique_ids(df, cfg.id_column),
                    "products": check_products(df, cfg.product_column, cfg.allowed_products),
                    "dates": check_dates(df, cfg.date_column, cfg.cutoff_date),
                    "no_empty_text": check_no_empty_text(df, cfg.text_column),
                    "subproduct_present": check_subproduct_present(df, cfg.subproduct_column),
                    "not_empty": check_not_empty(df),
                })

            status = all(passed for passed, _ in checks.values())
            report = {"validation_status": status,
                      "checks": {name: {"passed": p, "details": d} for name, (p, d) in checks.items()}}

            # Write the report FIRST, so a failure is always explained
            cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            failed = [name for name, (p, _) in checks.items() if not p]
            logger.info(f"Validation: {len(checks) - len(failed)}/{len(checks)} passed | report: {cfg.report_path}")

            if not status:
                raise ValueError(f"Validation failed: {failed} (see {cfg.report_path})")

            return DataValidationArtifact(validation_status=status, report_path=cfg.report_path, validated_parquet=cfg.input_parquet)

        except Exception as e:
            raise TicketRouterException(e) from e