from __future__ import annotations

import unittest
from pathlib import Path

from src.Donut.donut_train_logic import DEFAULT_DATASET_ROOT as DONUT_LOGIC_ROOT
from src.Donut.run_donut_training import DEFAULT_TRAINING_CONFIG as DONUT_CONFIG
from src.PP_parser.PPStructureV3_parser import DEFAULT_IMAGE_PATH as PP_IMAGE_PATH
from src.PP_parser.prepare_finetune import DEFAULT_DATASET_ROOT as PP_ROOT
from src.Qwen.qwen_finetune_logic import DEFAULT_DATASET_ROOT as QWEN_LOGIC_ROOT
from src.Qwen.run_qwen_training import DEFAULT_TRAINING_CONFIG as QWEN_CONFIG
from src.eval_suite import DEFAULT_TESTSET_PATH
from src.utils.annotation_audit import DEFAULT_DATASET_ROOT as AUDIT_ROOT
from src.utils.normalize_gross_weights import DEFAULT_DATASET_ROOT as NORMALIZE_ROOT
from src.utils.prediction_review import DEFAULT_DATASET_ROOT as REVIEW_ROOT


class DatasetDefaultTests(unittest.TestCase):
    def test_all_dataset_defaults_use_the_non_redundant_current_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected = repo_root / "data" / "datasets" / "250_CMRS_240dpi_20260707"
        defaults = {
            PP_ROOT,
            QWEN_LOGIC_ROOT,
            QWEN_CONFIG["dataset_root"],
            DONUT_LOGIC_ROOT,
            DONUT_CONFIG["dataset_root"],
            AUDIT_ROOT,
            repo_root / NORMALIZE_ROOT,
            REVIEW_ROOT,
        }

        self.assertEqual(defaults, {expected})
        self.assertTrue(PP_IMAGE_PATH.is_relative_to(expected))
        self.assertEqual(DEFAULT_TESTSET_PATH, expected / "test")


if __name__ == "__main__":
    unittest.main()
