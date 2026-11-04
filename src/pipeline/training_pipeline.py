"""Run all 6 stages in one go (convenience). DVC runs the same stage functions one process each: `dvc repro`."""
from src.pipeline.stages import (stage_01_ingest, stage_02_validate, stage_03_labels, stage_04_split,
                                 stage_05_train, stage_06_evaluate)

STAGES = [stage_01_ingest, stage_02_validate, stage_03_labels, stage_04_split, stage_05_train, stage_06_evaluate]


def run():
    return [stage.main() for stage in STAGES]


if __name__ == "__main__":
    run()
