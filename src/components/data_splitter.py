import pandas as pd

from src.entity.artifact_entity import DataSplitterArtifact
from src.entity.config_entity import DataSplitterConfig
from src.exception import TicketRouterException
from src.logger import logger


class DataSplitter:
    def __init__(self, config: DataSplitterConfig):
        self.config = config

    def initiate_data_splitting(self) -> DataSplitterArtifact:
        try:
            cfg = self.config
            df = pd.read_parquet(cfg.input_parquet)
            date = df[cfg.date_column]
            ts = pd.Timestamp                    # config dates are `date`; the column is datetime64

            # 1. Half-open split: start <= date < end_exclusive (a boundary day lands in exactly one split)
            train = df[date < ts(cfg.train_end)].copy()
            val = df[(date >= ts(cfg.val_start)) & (date < ts(cfg.val_end))].copy()
            test = df[(date >= ts(cfg.test_start)) & (date < ts(cfg.test_end))].copy()
            if len(train) + len(val) + len(test) != len(df):
                raise ValueError(f"Split lost or duplicated rows: {len(train)}+{len(val)}+{len(test)} != {len(df)}")
            logger.info(f"Split: train {len(train):,} | val {len(val):,} | test {len(test):,}")
            del df, date                      # the 3 copies hold everything; free ~1 GB before the heavy de-dup

            # 2. Train de-dup: one row per text, majority label wins.
            #    Tie-break RULE (not row order): most votes -> label A-Z -> lowest Complaint ID
            text, lab, cid = cfg.text_column, cfg.label_column, cfg.id_column
            n_train_full = len(train)
            train["n"] = train.groupby([text, lab])[lab].transform("count")

            pairs = train.groupby([text, lab]).size().reset_index(name="n")
            top = pairs[pairs["n"] == pairs.groupby(text)["n"].transform("max")]
            n_ties = int(top.groupby(text).size().gt(1).sum())       # texts with 2+ labels sharing the top count

            train = (train.sort_values(["n", lab, cid], ascending=[False, True, True])
                          .drop_duplicates(subset=[text], keep="first")
                          .drop(columns="n"))
            n_train_removed_dups = n_train_full - len(train)
            logger.info(f"Train de-dup: {len(train):,} texts | removed {n_train_removed_dups:,} | ties {n_ties}")

            # 3. test_dedup: remove texts seen in train, then exact copies (text + label).
            #    NO majority vote on test: test labels stay exactly as filed.
            #    (train after de-dup has the same SET of texts as before, so either one works here)
            seen = test[text].isin(train[text])
            n_test_seen_in_train = int(seen.sum())
            test_dedup = (test[~seen].sort_values(cid, kind="stable")          # rule: lowest ID survives
                                     .drop_duplicates(subset=[text, lab], keep="first"))
            logger.info(f"test_dedup: {len(test_dedup):,} | seen in train: {n_test_seen_in_train:,}")

            # 4. Shuffle train (otherwise it's saved sorted by vote count, the Day 10 bug), save, return
            train = train.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            parts = {"train": train, "val": val, "test": test, "test_dedup": test_dedup}
            paths = {name: cfg.output_dir / f"{name}.parquet" for name in parts}
            for name, part in parts.items():
                part.to_parquet(paths[name], index=False)
            logger.info(f"Saved 4 splits -> {cfg.output_dir}")

            return DataSplitterArtifact(
                train_parquet=paths["train"], val_parquet=paths["val"],
                test_parquet=paths["test"], test_dedup_parquet=paths["test_dedup"],
                n_train=len(train), n_val=len(val), n_test=len(test), n_test_dedup=len(test_dedup),
                n_train_removed_dups=n_train_removed_dups, n_ties=n_ties, n_test_seen_in_train=n_test_seen_in_train,
            )

        except Exception as e:
            raise TicketRouterException(e) from e
