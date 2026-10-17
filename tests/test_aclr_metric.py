import importlib.util
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "steps" / "final_evaluate_unified.py"
SPEC = importlib.util.spec_from_file_location("final_evaluate_unified", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class AclrMetricTest(unittest.TestCase):
    def test_minimum_point_uses_band_floor_values(self):
        freq_hz = np.array([-180.0, -120.0, 20.0, 80.0, 120.0, 180.0]) * 1e6
        error_psd_db = np.array([-62.0, -60.0, -20.0, -20.0, -70.0, -68.0])
        original_psd_db = np.array([-90.0, -90.0, -7.0, -5.0, -90.0, -90.0])

        lower, upper, worst = EVALUATOR.aclr_from_psd_minimum(
            freq_hz, error_psd_db, original_psd_db,
            (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0),
        )

        self.assertAlmostEqual(lower, -55.0)
        self.assertAlmostEqual(upper, -63.0)
        self.assertAlmostEqual(worst, -55.0)

    @unittest.skipUnless(
        (PROJECT_ROOT / "data" / "DPD_data.mat").exists(),
        "DPD_data.mat is not available",
    )
    def test_dataset_error_y_validation_reference(self):
        features, targets, dataset_errors, *_ = EVALUATOR.load_iq_arrays(
            PROJECT_ROOT / "data" / "DPD_data.mat", "rms"
        )
        split_idx = int(min(len(features), len(targets), len(dataset_errors)) * 0.8)
        target = EVALUATOR.tensor_to_complex(targets[split_idx:].T)
        dataset_error = EVALUATOR.tensor_to_complex(dataset_errors[split_idx:].T)
        freq_hz, target_psd_db = EVALUATOR.psd_db(target, 491.52e6, 4096)
        _, error_psd_db = EVALUATOR.psd_db(dataset_error, 491.52e6, 4096)

        point = EVALUATOR.aclr_from_psd_minimum(
            freq_hz, error_psd_db, target_psd_db,
            (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0),
        )
        self.assertAlmostEqual(point[2], -45.1045215484331, places=6)


if __name__ == "__main__":
    unittest.main()
