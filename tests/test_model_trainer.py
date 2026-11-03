import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import get_trainer_config
from src.exception import TicketRouterException
from src.utils import sha256_file


def make_config(tmp_path, **overrides):
    real = get_trainer_config(tmp_path / "train.parquet")
    rows = [(f"topic{i} problem number {k}", label) for i, label in enumerate(real.labels) for k in range(2)]
    pd.DataFrame(rows, columns=[real.text_column, real.label_column]).to_parquet(real.train_parquet, index=False)
    return dataclasses.replace(real, output_dir=tmp_path / "model", min_df=1, **overrides)


def test_trains_and_saves(tmp_path):
    cfg = make_config(tmp_path)
    art = ModelTrainer(cfg).initiate_model_trainer()
    for path, sha in [(art.tfidf_path, art.tfidf_sha256), (art.logreg_path, art.logreg_sha256),
                      (art.label_order_path, art.label_order_sha256)]:
        assert path.exists() and sha == sha256_file(path)
    assert json.loads(art.label_order_path.read_text(encoding="utf-8")) == list(cfg.labels)
    assert art.n_train == 22                       # 11 labels x 2 rows


def test_wrong_label_order_raises(tmp_path):
    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(cfg, labels=tuple(reversed(cfg.labels)))
    with pytest.raises(TicketRouterException) as e:
        ModelTrainer(cfg).initiate_model_trainer()
    assert "Label order mismatch" in str(e.value)
