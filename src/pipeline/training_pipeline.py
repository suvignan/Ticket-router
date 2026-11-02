from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.label_builder import LabelBuilder
from src.entity.config_entity import get_ingestion_config, get_label_builder_config, get_validation_config
from src.logger import logger


def run():
    ingestion_artifact = DataIngestion(get_ingestion_config()).initiate_data_ingestion()
    logger.info(f"Ingestion artifact: {ingestion_artifact}")

    validation_config = get_validation_config(ingestion_artifact.output_parquet)      # the baton
    validation_artifact = DataValidation(validation_config).initiate_data_validation()
    logger.info(f"Validation artifact: {validation_artifact}")

    label_config = get_label_builder_config(validation_artifact.validated_parquet)
    label_artifact = LabelBuilder(label_config).initiate_label_building()
    logger.info(f"Label artifact: {label_artifact}")
    return label_artifact, validation_artifact, ingestion_artifact


if __name__ == "__main__":
    run()
