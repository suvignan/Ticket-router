from src.components.model_trainer import ModelTrainer
from src.constants import ARTIFACT_DIR, SPLIT_DIR
from src.entity.config_entity import get_trainer_config
from src.logger import logger
from src.utils import save_artifact


def main():
    artifact = ModelTrainer(get_trainer_config(SPLIT_DIR / "train.parquet")).initiate_model_trainer()
    save_artifact(artifact, ARTIFACT_DIR / "train.json")
    logger.info(f"train artifact: {artifact}")
    return artifact


if __name__ == "__main__":
    main()
