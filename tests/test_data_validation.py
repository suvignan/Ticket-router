import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.components.data_validation import (check_dates, check_products, check_unique_ids, check_columns,
                                            check_dtypes, check_no_empty_text, check_subproduct_present, check_not_empty)
from src.entity.config_entity import get_validation_config

CFG = get_validation_config(Path("unused.parquet"))      # real columns/products/cutoff; the path isn't read


def clean_df():
    return pd.DataFrame({
        "Consumer complaint narrative": ["I was charged twice", "My loan was sold", "Wrong balance"],
        "Product": ["Credit card", "Mortgage", "Debt collection"],
        "Date received": pd.to_datetime(["2024-03-15", "2024-04-01", "2025-05-31"]),
        "Complaint ID": [1, 2, 3],
        "Sub-product": ["Store credit card", "FHA mortgage", "I do not know"],
    })


def test_clean_df_passes_all():
    df = clean_df()
    results = [check_columns(df, CFG.columns), check_dtypes(df, CFG.date_column, CFG.id_column),
               check_unique_ids(df, CFG.id_column), check_products(df, CFG.product_column, CFG.allowed_products),
               check_dates(df, CFG.date_column, CFG.cutoff_date), check_no_empty_text(df, CFG.text_column),
               check_subproduct_present(df, CFG.subproduct_column), check_not_empty(df)]
    assert all(passed for passed, _ in results)


def test_unknown_product_is_listed():
    df = clean_df()
    df.loc[0, "Product"] = "Credit card (new)"
    passed, details = check_products(df, CFG.product_column, CFG.allowed_products)
    assert not passed
    assert details["unexpected"] == ["Credit card (new)"]


def test_duplicate_id_fails():
    df = clean_df()
    df.loc[1, "Complaint ID"] = df.loc[0, "Complaint ID"]
    passed, details = check_unique_ids(df, CFG.id_column)
    assert not passed and details["n_duplicates"] == 2


def test_time_part_fails():
    df = clean_df()
    df.loc[0, "Date received"] = pd.Timestamp("2024-03-15 10:30:00")
    passed, details = check_dates(df, CFG.date_column, CFG.cutoff_date)
    assert not passed and details["n_with_time_part"] == 1

def test_missing_product_fails():
    df = clean_df()
    df.loc[0, "Product"] = None
    passed, details = check_products(df, CFG.product_column, CFG.allowed_products)
    assert not passed and details["n_missing"] == 1
    assert details["unexpected"] == []  # missing is not "unexpected"
