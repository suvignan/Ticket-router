from src.components.data_validation import DataValidation
from src.constants import ARTIFACT_DIR, INGESTED_PARQUET
from src.entity.config_entity import get_validation_config
from src.logger import logger
from src.utils import save_artifact


def main():
    artifact = DataValidation(get_validation_config(INGESTED_PARQUET)).initiate_data_validation()
    save_artifact(artifact, ARTIFACT_DIR / "validation.json")
    logger.info(f"validation artifact: {artifact}")
    return artifact


if __name__ == "__main__":
    main()
