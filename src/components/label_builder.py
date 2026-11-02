import pandas as pd

from src.entity.artifact_entity import LabelBuilderArtifact
from src.entity.config_entity import LabelBuilderConfig
from src.exception import TicketRouterException
from src.logger import logger


class LabelBuilder:
    def __init__(self, config: LabelBuilderConfig):
        self.config = config

    def initiate_label_building(self) -> LabelBuilderArtifact:
        try:
            cfg = self.config
            df = pd.read_parquet(cfg.input_parquet)
            lab, sub = cfg.label_column, cfg.subproduct_column

            # 1. Copy: Product stays untouched for traceability
            df[lab] = df[cfg.product_column]

            # 2. Renames (count BEFORE replacing: afterwards the old names are gone)
            rename = dict(cfg.rename_map)
            n_renamed = int(df[lab].isin(rename.keys()).sum())
            df[lab] = df[lab].replace(rename)
            leftover = set(rename) & set(df[lab].unique())
            if leftover:
                raise ValueError(f"Rename failed, still present: {sorted(leftover)}")
            logger.info(f"Renamed: {n_renamed:,}")

            # 3. Card split: two EXPLICIT masks, then nothing may stay in the old box
            old_card = df[lab] == cfg.old_card_product
            to_credit = old_card & df[sub].isin(cfg.credit_subproducts)
            to_prepaid = old_card & df[sub].isin(cfg.prepaid_subproducts)
            n_card_to_credit, n_card_to_prepaid = int(to_credit.sum()), int(to_prepaid.sum())
            df.loc[to_credit, lab] = cfg.credit_label
            df.loc[to_prepaid, lab] = cfg.prepaid_label
            unmapped = df.loc[df[lab] == cfg.old_card_product, sub]
            if len(unmapped):
                raise ValueError(f"{len(unmapped)} card rows with unmapped sub-products: "
                                 f"{unmapped.value_counts(dropna=False).to_dict()}")
            logger.info(f"Card split: {n_card_to_credit:,} -> credit | {n_card_to_prepaid:,} -> prepaid")

            # 4. Credit-repair move: AFTER renames, because from_label is the NEW credit-reporting name
            repair = (df[lab] == cfg.repair_from_label) & (df[sub] == cfg.repair_sub_product)
            n_credit_repair_moved = int(repair.sum())
            df.loc[repair, lab] = cfg.repair_to_label
            logger.info(f"Credit-repair moved: {n_credit_repair_moved:,}")

            # 5. Final check: exactly the allowed labels, no extras, none missing
            present = set(df[lab].unique())
            extra = sorted(map(str, present - set(cfg.labels)))      # str(): a NaN can't be sorted with strings
            missing = sorted(set(cfg.labels) - present)
            if extra or missing:
                raise ValueError(f"Label set wrong | extra: {extra} | missing: {missing}")

            # 6. Keep only the KEEP columns (drops Sub-product: never a feature), save, return
            df = df[list(cfg.keep_columns)]
            cfg.output_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cfg.output_parquet, index=False)
            label_counts = {k: int(v) for k, v in df[lab].value_counts().to_dict().items()}
            logger.info(f"Saved {len(df):,} labelled rows -> {cfg.output_parquet}")

            return LabelBuilderArtifact(
                output_parquet=cfg.output_parquet, n_rows=len(df),
                n_renamed=n_renamed, n_card_to_credit=n_card_to_credit, n_card_to_prepaid=n_card_to_prepaid,
                n_credit_repair_moved=n_credit_repair_moved, label_counts=label_counts,
            )

        except Exception as e:
            raise TicketRouterException(e) from e
