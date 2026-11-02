from src.components.data_ingestion import DataIngestion, DataIngestionArtifact
from src.entity.config_entity import get_ingestion_config
from src.logger import logger


def run():
    config = get_ingestion_config()                                  # builds the recipe card from the YAML files
    artifact = DataIngestion(config).initiate_data_ingestion()            # run the machine
    logger.info(f"Ingestion artifact: {artifact}")
    return artifact


if __name__ == "__main__":
    run()
