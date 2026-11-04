from src.components.data_splitter import DataSplitter
from src.constants import ARTIFACT_DIR, LABELED_PARQUET
from src.entity.config_entity import get_splitter_config
from src.logger import logger
from src.utils import save_artifact


def main():
    artifact = DataSplitter(get_splitter_config(LABELED_PARQUET)).initiate_data_splitting()
    save_artifact(artifact, ARTIFACT_DIR / "split.json")
    logger.info(f"split artifact: {artifact}")
    return artifact


if __name__ == "__main__":
    main()
