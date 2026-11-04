from src.components.data_ingestion import DataIngestion
from src.constants import ARTIFACT_DIR
from src.entity.config_entity import get_ingestion_config
from src.logger import logger
from src.utils import save_artifact


def main():
    artifact = DataIngestion(get_ingestion_config()).initiate_data_ingestion()
    save_artifact(artifact, ARTIFACT_DIR / "ingestion.json")
    logger.info(f"ingestion artifact: {artifact}")
    return artifact


if __name__ == "__main__":
    main()
