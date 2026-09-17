"""Tests for auditable train/test protocol metadata."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_experiment import DEFAULTS, _distribution_metadata


class BOExperimentMetadataTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(DEFAULTS)

    def test_checkpoint_condition_is_labeled_matched(self):
        metadata = {
            "checkpoint_id": "/model/run",
            "training_config": {
                "data_kwargs": {"rho": 0.6},
                "task_kwargs": {"snr": 2.0, "noise_rho": 0.3},
            },
        }
        self.config["noise_rho"] = 0.3
        result = _distribution_metadata(metadata, self.config, rho=0.6, snr=2.0)
        self.assertEqual(result["evaluation_protocol"], "matched")
        self.assertEqual(result["train_rho_x"], result["test_rho_x"])
        self.assertEqual(result["train_rho_e"], result["test_rho_e"])

    def test_any_distribution_difference_is_labeled_shift(self):
        metadata = {
            "checkpoint_id": "/model/run",
            "training_config": {
                "data_kwargs": {"rho": 0.6},
                "task_kwargs": {"snr": 2.0, "noise_rho": 0.0},
            },
        }
        result = _distribution_metadata(metadata, self.config, rho=0.9, snr=2.0)
        self.assertEqual(result["evaluation_protocol"], "shift")

    def test_snr_sweep_does_not_relabel_matched_dependence_as_shift(self):
        metadata = {
            "checkpoint_id": "/model/run",
            "training_config": {
                "data_kwargs": {"rho": 0.6},
                "task_kwargs": {"snr": 2.0, "noise_rho": 0.0},
            },
        }
        result = _distribution_metadata(metadata, self.config, rho=0.6, snr=0.1)
        self.assertEqual(result["protocol"], "matched")
        self.assertEqual(result["train_snr"], 2.0)
        self.assertEqual(result["test_snr"], 0.1)

    def test_change_point_schedule_participates_in_match_decision(self):
        metadata = {
            "checkpoint_id": "/model/run",
            "training_config": {
                "data_kwargs": {"rho": 0.0, "rho_after": 0.9, "change_point": 10},
                "task_kwargs": {"snr": 2.0, "noise_rho": 0.0},
            },
        }
        self.config["feature_rho_after"] = 0.9
        self.config["feature_change_point"] = 11
        result = _distribution_metadata(metadata, self.config, rho=0.0, snr=2.0)
        self.assertEqual(result["evaluation_protocol"], "shift")

    def test_classical_estimator_is_condition_local_and_matched(self):
        result = _distribution_metadata({"checkpoint_id": None}, self.config, rho=0.9, snr=0.8)
        self.assertEqual(result["evaluation_protocol"], "matched")
        self.assertEqual(result["train_distribution"]["source"], "condition_local_estimator")


if __name__ == "__main__":
    unittest.main()
