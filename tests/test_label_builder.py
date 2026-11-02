import dataclasses
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.components.label_builder import LabelBuilder
from src.entity.config_entity import get_label_builder_config
from src.exception import TicketRouterException

OLD_CARD = "Credit card or prepaid card"
OLD_REPORTING = "Credit reporting, credit repair services, or other personal consumer reports"
PAYDAY_OLD = "Payday loan, title loan, or personal loan"
PAYDAY_NEW = "Payday loan, title loan, personal loan, or advance loan"

ROWS = [  # (Product, Sub-product, expected label)
    (PAYDAY_OLD,    "Payday loan",            PAYDAY_NEW),                   # rename
    (OLD_CARD,      "Store credit card",      "Credit card"),                # card -> credit
    (OLD_CARD,      "Gift card",              "Prepaid card"),               # card -> prepaid
    (OLD_REPORTING, "Credit repair services", "Debt or credit management"),  # rename THEN move
    ("Mortgage",    "FHA mortgage",           "Mortgage"),                   # untouched
]


def run_builder(tmp_path, rows):
    df = pd.DataFrame({
        "Complaint ID": range(1, len(rows) + 1),
        "Date received": pd.Timestamp("2024-03-15"),
        "Consumer complaint narrative": "some text",
        "Product": [r[0] for r in rows],
        "Sub-product": [r[1] for r in rows],
    })
    src = tmp_path / "in.parquet"
    df.to_parquet(src, index=False)
    expected = {r[2] for r in ROWS}                    # only the labels this tiny world should have
    cfg = dataclasses.replace(get_label_builder_config(src),
                              output_parquet=tmp_path / "out.parquet", labels=tuple(sorted(expected)))
    return LabelBuilder(cfg).initiate_label_building()


def test_each_rule_once(tmp_path):
    art = run_builder(tmp_path, ROWS)
    out = pd.read_parquet(art.output_parquet)
    assert out["label"].tolist() == [r[2] for r in ROWS]
    assert (art.n_renamed, art.n_card_to_credit, art.n_card_to_prepaid, art.n_credit_repair_moved) == (2, 1, 1, 1)   # rows 1 AND 4 are renamed
    assert "Sub-product" not in out.columns   # dropped: never a feature


def test_unknown_card_subproduct_raises(tmp_path):
    rows = ROWS + [(OLD_CARD, "Crypto card", None)]
    with pytest.raises(TicketRouterException) as e:
        run_builder(tmp_path, rows)
    assert "Crypto card" in str(e.value)           # the error must NAME the unknown sub-product
