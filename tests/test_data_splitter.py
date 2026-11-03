import dataclasses
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.components.data_splitter import DataSplitter
from src.entity.config_entity import get_splitter_config

CC, MORT, VEH, DEBT = "Credit card", "Mortgage", "Vehicle loan or lease", "Debt collection"
ROWS = [  # (Complaint ID, text, label, date)
    (1, "A", CC, "2024-05-01"), (2, "A", CC, "2024-06-01"), (3, "A", CC, "2024-07-01"), (4, "A", MORT, "2024-08-01"),
    (5, "B", VEH, "2024-02-01"), (6, "B", VEH, "2024-03-01"), (7, "B", CC, "2024-04-01"), (8, "B", CC, "2024-05-01"),
    (9, "C", MORT, "2025-01-01"),
    (10, "A", DEBT, "2025-03-10"),
    (11, "D", MORT, "2025-04-01"), (12, "D", MORT, "2025-04-02"), (13, "D", CC, "2025-04-03"),
]


def run_splitter(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["Complaint ID", "Consumer complaint narrative", "label", "Date received"])
    df["Date received"] = pd.to_datetime(df["Date received"])
    df["Product"] = df["label"]                               # LabelBuilder's output has this column too
    tmp_path.mkdir(parents=True, exist_ok=True)              # sub-folders "a"/"b" don't exist yet
    src = tmp_path / "in.parquet"
    df.to_parquet(src, index=False)
    cfg = dataclasses.replace(get_splitter_config(src), output_dir=tmp_path / "splits")
    art = DataSplitter(cfg).initiate_data_splitting()
    train = pd.read_parquet(art.train_parquet)
    winners = dict(zip(train["Consumer complaint narrative"], train["label"]))
    survivors = set(train["Complaint ID"])
    test_dedup_ids = set(pd.read_parquet(art.test_dedup_parquet)["Complaint ID"])
    return art, winners, survivors, test_dedup_ids


def test_split_rules(tmp_path):
    art, winners, survivors, test_dedup_ids = run_splitter(tmp_path, ROWS)
    assert (art.n_train, art.n_val, art.n_test) == (2, 1, 4)      # train counted AFTER de-dup (texts A, B)
    assert winners == {"A": CC, "B": CC}                           # A: 3 vs 1 majority; B: 2 vs 2 tie -> A-Z
    assert survivors == {1, 7}                                     # lowest ID among each WINNING label's rows
    assert art.n_ties == 1 and art.n_train_removed_dups == 6       # 8 train rows -> 2 texts
    assert art.n_test_seen_in_train == 1                           # row 10: text A is in train
    assert test_dedup_ids == {11, 13}                              # 12 = copy of 11; 13 has a different label


def test_row_order_does_not_matter(tmp_path):
    first = run_splitter(tmp_path / "a", ROWS)[1:]
    second = run_splitter(tmp_path / "b", list(reversed(ROWS)))[1:]
    assert first == second
