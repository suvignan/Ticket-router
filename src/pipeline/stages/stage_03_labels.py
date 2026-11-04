from src.components.label_builder import LabelBuilder
from src.constants import ARTIFACT_DIR, INGESTED_PARQUET
from src.entity.config_entity import get_label_builder_config
from src.logger import logger
from src.utils import save_artifact


def main():
    artifact = LabelBuilder(get_label_builder_config(INGESTED_PARQUET)).initiate_label_building()   # validation never copies: same file
    save_artifact(artifact, ARTIFACT_DIR / "labels.json")
    logger.info(f"labels artifact: {artifact}")
    return artifact


if __name__ == "__main__":
    main()
