from src.components.data_ingestion import DataIngestion
from src.components.data_splitter import DataSplitter
from src.components.data_validation import DataValidation
from src.components.label_builder import LabelBuilder
from src.entity.config_entity import get_ingestion_config, get_label_builder_config, get_splitter_config, get_trainer_config, get_validation_config
from src.components.model_trainer import ModelTrainer       
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

    splitter_config = get_splitter_config(label_artifact.output_parquet)
    splitter_artifact = DataSplitter(splitter_config).initiate_data_splitting()
    logger.info(f"Splitter artifact: {splitter_artifact}")

    trainer_config = get_trainer_config(splitter_artifact.train_parquet)
    trainer_artifact = ModelTrainer(trainer_config).initiate_model_trainer()
    logger.info(f"Trainer artifact: {trainer_artifact}")
    return trainer_artifact, splitter_artifact, label_artifact, validation_artifact, ingestion_artifact


if __name__ == "__main__":
    run()
