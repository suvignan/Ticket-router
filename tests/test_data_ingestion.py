import dataclasses
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import get_ingestion_config
from src.exception import TicketRouterException
from src.utils import sha256_file


def write_fake_csv(path):
    # Each row tests exactly one rule, so a broken rule changes exactly one count
    path.write_text(
        "Date received,Product,Sub-product,Consumer complaint narrative,Complaint ID\n"
        "03/15/24,Credit card,Store credit card,I was charged twice for one purchase,1\n"      # kept
        "03/15/24,Credit card,Store credit card,,2\n"                                          # junk A: NaN
        "03/15/24,Credit card,Store credit card,XXXX XX/XX/XXXX,3\n"                           # junk B: redaction only
        "03/15/24,Credit card,Store credit card,See attached.,4\n"                             # junk C: short pointer
        "06/10/25,Credit card,Store credit card,My card was blocked without notice,5\n"        # dropped: after cutoff
        "03/15/24,Credit card,Store credit card,Please see attached letter about the wrong late fee,6\n"  # kept: > 30 chars
        '03/15/24,Credit card,Store credit card,"An unfinished narrative,7',                   # bad line: truncated, no \n
        encoding="utf-8",
    )


def make_config(tmp_path,**overrides):
    csv = tmp_path / "fake.csv"
    write_fake_csv(csv)
    real = get_ingestion_config()     # real rules from the YAML files
    # Swap ONLY the paths; output must live in tmp_path, never in data/interim/
    return dataclasses.replace(real, raw_csv=csv, output_parquet=tmp_path / "out.parquet", **overrides)


def test_ingestion(tmp_path):
    cfg = make_config(tmp_path)
    art = DataIngestion(cfg).initiate_data_ingestion()
    assert art.n_rows_read == 6
    assert art.n_bad_lines == 1
    assert art.n_junk_removed == 3
    assert art.n_after_cutoff == 2
    assert art.n_after_cutoff == len(pd.read_parquet(art.output_parquet))
    assert art.raw_sha256 == sha256_file(cfg.raw_csv)
    saved = pd.read_parquet(art.output_parquet)
    assert set(saved['Complaint ID']) == {1, 6}


def test_too_many_bad_lines(tmp_path):
    cfg = make_config(tmp_path, max_bad_lines=0)
    with pytest.raises(TicketRouterException) as e:
        DataIngestion(cfg).initiate_data_ingestion()
    assert "bad lines" in str(e.value)