import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import TicketRouterException
from src.logger import logger
from src.text import clean_text
from src.utils import sha256_file


def decide_gate(challenger: dict, champion: dict | None, cfg: ModelEvaluationConfig) -> tuple[bool, list[str]]:
    """Pure function: numbers in, verdict out. No files.
    challenger / champion = {"macro_f1": 0.69, "per_class": {"Prepaid card": 0.71, ...}}"""
    reasons = []

    # Rule 1: absolute floor (always applies, even with no champion)
    if challenger["macro_f1"] < cfg.min_macro_f1:
        reasons.append(f"macro-F1 {challenger['macro_f1']:.4f} below floor {cfg.min_macro_f1}")

    if champion is None:
        return len(reasons) == 0, reasons          # brand-new deployment: floor only

    # Rule 2: relative tolerance. Boundary: a drop of EXACTLY 0.005 passes (rule is challenger >= champion - T), so fail only on '>'
    drop = round(champion["macro_f1"] - challenger["macro_f1"], 6)
    if drop > cfg.max_macro_f1_drop:
        reasons.append(f"macro-F1 dropped {drop:.4f} vs champion (> {cfg.max_macro_f1_drop})")

    # Rule 3: each small class (check ALL of them, so every failure gets reported)
    for cls in cfg.small_classes:
        cls_drop = round(champion["per_class"][cls] - challenger["per_class"][cls], 6)
        if cls_drop > cfg.max_small_class_drop:
            reasons.append(f"{cls} dropped {cls_drop:.3f} (> {cfg.max_small_class_drop})")

    return len(reasons) == 0, reasons


def score(tfidf_path, logreg_path, texts, y, labels) -> dict:
    """Each model through ITS OWN vectoriser."""
    vec, model = joblib.load(tfidf_path), joblib.load(logreg_path)
    if list(model.classes_) != list(labels):
        raise ValueError(f"Label order mismatch in {logreg_path}")
    pred = model.predict(vec.transform(texts))
    per_class = f1_score(y, pred, average=None, labels=list(labels))    # array, in labels order
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred)),
            "per_class": {label: float(f) for label, f in zip(labels, per_class)}}


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            cfg, champ = self.config, self.config.champion

            # 1. Right champion file? (checksum = "right file", not "same model")
            if champ is not None:
                for path, expected in [(champ.tfidf_path, champ.tfidf_sha256), (champ.logreg_path, champ.logreg_sha256),
                                       (champ.label_order_path, champ.label_order_sha256)]:
                    if sha256_file(path) != expected:
                        raise ValueError(f"Champion checksum mismatch: {path}")

            # 2. Load val, clean ONCE (both models get the same texts)
            val = pd.read_parquet(cfg.val_parquet, columns=[cfg.text_column, cfg.label_column])
            texts, y = val[cfg.text_column].map(clean_text), val[cfg.label_column]

            # 3. Score both
            challenger = score(cfg.challenger_tfidf, cfg.challenger_logreg, texts, y, cfg.labels)
            champion = None if champ is None else score(champ.tfidf_path, champ.logreg_path, texts, y, cfg.labels)

            # 4. Canary: a frozen model's score must never move
            if champion is not None:
                drift = abs(champion["macro_f1"] - champ.recorded_val_macro_f1)
                if drift > cfg.max_champion_drift:
                    raise ValueError(f"Environment changed: champion val macro-F1 {champion['macro_f1']:.6f} "
                                     f"vs recorded {champ.recorded_val_macro_f1} (drift {drift:.6f})")

            # 5. Decide
            passed, reasons = decide_gate(challenger, champion, cfg)
            diff = None if champion is None else round(challenger["macro_f1"] - champion["macro_f1"], 6)
            per_class_diff = None if champion is None else {
                c: round(challenger["per_class"][c] - champion["per_class"][c], 6) for c in cfg.labels}

            # 6. Report FIRST, raise AFTER (so a failed gate still leaves evidence)
            cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
            report = {"gate_passed": passed, "reasons": reasons, "challenger": challenger,
                      "champion": champion, "macro_f1_diff": diff, "per_class_f1_diff": per_class_diff}
            cfg.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info(f"Gate {'PASSED' if passed else 'FAILED'} | challenger {challenger['macro_f1']:.4f} | diff {diff}")
            # No raise here: the stage logs the verdict to MLflow FIRST (failures are evidence), then raises

            return ModelEvaluationArtifact(
                gate_passed=passed, reasons=tuple(reasons),
                challenger_macro_f1=challenger["macro_f1"],
                champion_macro_f1=None if champion is None else champion["macro_f1"],
                macro_f1_diff=diff, per_class_f1_diff=per_class_diff, report_path=cfg.report_path)

        except Exception as e:
            raise TicketRouterException(e) from e