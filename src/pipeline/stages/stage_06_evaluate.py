import os
import subprocess
import sys

from dotenv import load_dotenv

from src.components.model_evaluation import ModelEvaluation
from src.constants import ARTIFACT_DIR, MODEL_DIR, PARAMS_FILE, ROOT, SPLIT_DIR
from src.entity.artifact_entity import DataIngestionArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig, get_evaluation_config
from src.logger import logger
from src.utils import flatten, load_artifact, read_yaml, save_artifact, slugify


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def log_to_mlflow(artifact: ModelEvaluationArtifact, cfg: ModelEvaluationConfig) -> None:
    """One MLflow run per evaluation, passed OR failed: a rejected challenger is evidence too."""
    load_dotenv(ROOT / ".env")                       # MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD
    if os.environ.get("MLFLOW_DISABLED") == "1":     # tests / CI without credentials
        logger.info("MLflow logging SKIPPED (MLFLOW_DISABLED=1)")
        return
    missing = [v for v in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD") if not os.environ.get(v)]
    if missing:                                      # a real run must never silently lose its audit trail
        raise RuntimeError(f"MLflow credentials missing: {missing}. Set them in .env, or MLFLOW_DISABLED=1 to skip on purpose")

    import mlflow                                    # imported here: MLflow is only needed when logging
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # MLflow prints an emoji; the Windows console (cp1252) crashes on it

    params = read_yaml(PARAMS_FILE)
    report = read_yaml(cfg.report_path)              # JSON is valid YAML
    commit = git("rev-parse", "HEAD")
    metrics = {"challenger.macro_f1": artifact.challenger_macro_f1,
               "challenger.accuracy": report["challenger"]["accuracy"]}
    if artifact.champion_macro_f1 is not None:
        metrics |= {"champion.macro_f1": artifact.champion_macro_f1,
                    "champion.accuracy": report["champion"]["accuracy"],
                    "macro_f1_diff": artifact.macro_f1_diff}
    metrics |= {f"f1.{slugify(c)}": f for c, f in report["challenger"]["per_class"].items()}
    tags = {"gate_passed": str(artifact.gate_passed).lower(),
            "git_commit": commit,
            "git_dirty": str(bool(git("status", "--porcelain"))).lower(),   # uncommitted code = not reproducible from the commit
            "raw_sha256": load_artifact(DataIngestionArtifact, ARTIFACT_DIR / "ingestion.json").raw_sha256,
            "champion_logreg_sha256": cfg.champion.logreg_sha256 if cfg.champion else "none",
            "reasons": "; ".join(artifact.reasons) or "none"}

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("ticket-router-lr")
    with mlflow.start_run(run_name=f"lr-{commit[:7]}"):
        mlflow.log_params(flatten(params["model_trainer"], "model_trainer.") | flatten(params["model_evaluation"], "model_evaluation."))
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(cfg.report_path))
        mlflow.log_artifact(str(MODEL_DIR / "label_order.json"))
    logger.info(f"MLflow run logged | gate_passed={artifact.gate_passed} | commit {commit[:7]}")


def main():
    cfg = get_evaluation_config(SPLIT_DIR / "val.parquet", MODEL_DIR / "tfidf.joblib", MODEL_DIR / "logreg.joblib")
    artifact = ModelEvaluation(cfg).initiate_model_evaluation()   # writes the report, returns the verdict
    save_artifact(artifact, ARTIFACT_DIR / "evaluation.json")
    logger.info(f"evaluation artifact: {artifact}")
    log_to_mlflow(artifact, cfg)                                 # 1. evidence first ...
    if not artifact.gate_passed:                                 # 2. ... then stop the pipeline
        raise SystemExit(f"Model gate FAILED: {list(artifact.reasons)}")
    return artifact


if __name__ == "__main__":
    main()
