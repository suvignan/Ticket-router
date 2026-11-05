import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import TicketRouterException
from src.logger import logger
from src.text import clean_text
from src.utils import sha256_file


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            cfg = self.config
            # 1. Only the 2 columns the model uses (memory!)
            train = pd.read_parquet(cfg.train_parquet, columns=[cfg.text_column, cfg.label_column])

            # 2. Same clean_text as the API: one recipe for training AND serving
            texts = train[cfg.text_column].map(clean_text)
            y = train[cfg.label_column]

            # 3. Fit TF-IDF + LR (Config B-none)
            start = time.perf_counter()
            vec = TfidfVectorizer(min_df=cfg.min_df, ngram_range=cfg.ngram_range, sublinear_tf=cfg.sublinear_tf)
            X = vec.fit_transform(texts)
            model = LogisticRegression(max_iter=cfg.max_iter, C=cfg.C, class_weight=cfg.class_weight)
            model.fit(X, y)
            train_seconds = time.perf_counter() - start
            vocab_size = len(vec.vocabulary_)
            logger.info(f"Trained on {len(train):,} rows | vocab {vocab_size:,} | {train_seconds / 60:.1f} min")

            # 4. Label order must equal label_order.json (the Day 9 bug, caught automatically)
            if list(model.classes_) != list(cfg.labels):
                raise ValueError(f"Label order mismatch: model {list(model.classes_)} vs schema {list(cfg.labels)}")

            # 5. Save the 3 files together (one unit: model + its label order) + checksums
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            paths = {"tfidf": cfg.output_dir / "tfidf.joblib",
                     "logreg": cfg.output_dir / "logreg.joblib",
                     "label_order": cfg.output_dir / "label_order.json"}
            joblib.dump(vec, paths["tfidf"])
            joblib.dump(model, paths["logreg"])
            # newline="\n": same bytes on Windows and Linux (CI/Docker); default write_text writes CRLF on Windows
            paths["label_order"].write_text(json.dumps(list(model.classes_), indent=2), encoding="utf-8", newline="\n")
            sha = {name: sha256_file(path) for name, path in paths.items()}
            logger.info(f"Saved model -> {cfg.output_dir} | logreg sha256 {sha['logreg'][:12]}")

            return ModelTrainerArtifact(
                tfidf_path=paths["tfidf"], logreg_path=paths["logreg"], label_order_path=paths["label_order"],
                tfidf_sha256=sha["tfidf"], logreg_sha256=sha["logreg"], label_order_sha256=sha["label_order"],
                vocab_size=vocab_size, n_train=len(train), train_seconds=round(train_seconds, 1),
            )

        except Exception as e:
            raise TicketRouterException(e) from e
