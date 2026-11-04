from src.components.model_evaluation import ModelEvaluation
from src.constants import ARTIFACT_DIR, MODEL_DIR, SPLIT_DIR
from src.entity.config_entity import get_evaluation_config
from src.logger import logger
from src.utils import save_artifact


def main():
    cfg = get_evaluation_config(SPLIT_DIR / "val.parquet", MODEL_DIR / "tfidf.joblib", MODEL_DIR / "logreg.joblib")
    artifact = ModelEvaluation(cfg).initiate_model_evaluation()
    save_artifact(artifact, ARTIFACT_DIR / "evaluation.json")
    logger.info(f"evaluation artifact: {artifact}")
    return artifact


if __name__ == "__main__":
    main()
