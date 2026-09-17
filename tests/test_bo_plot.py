"""Tests for phase-boundary and hierarchical uncertainty summaries."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_plot import (
    critical_snr_brackets,
    load_groups,
    propose_snr_boundary,
    summarize_training_replicates,
)


def _record(value):
    return {"metrics": {"bo_candidate": {"mean": value}}}


def _group(checkpoint, training_seed, low_values, high_values):
    return {
        "identity": {
            "protocol_id": "protocol",
            "model": "gpt2_embd=64_layer=3_head=2",
            "checkpoint_id": checkpoint,
            "training_seed": training_seed,
            "d": 8,
        },
        "protocol": {"linear_probe": False},
        "cells": {
            (0.6, 1.0, 8): {i: _record(value) for i, value in enumerate(low_values)},
            (0.6, 2.0, 8): {i: _record(value) for i, value in enumerate(high_values)},
        },
    }


class BOPlotTests(unittest.TestCase):
    def test_missing_cell_is_not_bridged(self):
        result = critical_snr_brackets([(1, 0.2), (2, None), (4, 0.8)], target=0.5)
        self.assertEqual(result["status"], "no_observed_crossing")
        self.assertEqual(result["brackets"], [])

    def test_reports_every_observed_direction_without_unique_estimate(self):
        result = critical_snr_brackets([(1, 0.2), (2, 0.8), (4, 0.1)], target=0.5)
        self.assertEqual(len(result["brackets"]), 2)
        self.assertEqual(result["brackets"][0]["direction"], "upward")
        self.assertEqual(result["brackets"][1]["direction"], "downward")
        self.assertFalse(result["unique_critical_snr_estimated"])

    def test_boundary_proposal_refines_every_nonmonotonic_crossing(self):
        result = propose_snr_boundary([(1, 0.2), (2, 0.8), (8, 0.1)], target=0.5)
        self.assertEqual(result["action"], "refine_all_observed_crossings")
        self.assertEqual(len(result["brackets"]), 2)
        self.assertEqual(result["suggested_snrs"], [2 ** 0.5, 4.0])

    def test_boundary_proposal_expands_from_closest_edge_when_no_crossing(self):
        result = propose_snr_boundary([(1, 0.1), (2, 0.2), (4, 0.4)], target=0.5)
        self.assertEqual(result["action"], "expand_from_closest_edge")
        self.assertEqual(result["closest_sampled_points"][0]["snr"], 4.0)
        self.assertEqual(result["suggested_snrs"], [8.0])

    def test_legacy_and_metadata_aware_outputs_cannot_be_merged(self):
        import json
        import tempfile

        legacy_record = {
            "protocol_id": "p", "model": "ols", "checkpoint_id": None,
            "training_seed": None, "seed": 1, "d": 2, "rho": 0.0,
            "snr": 1.0, "k": 2, "metrics": {},
        }
        aware_record = {
            **legacy_record,
            "train_rho_x": 0.0, "test_rho_x": 0.0,
            "train_rho_e": 0.0, "test_rho_e": 0.0,
            "protocol": "matched", "evaluation_protocol": "matched", "train_seed": None,
            "eval_seed": 1, "train_snr": 1.0, "test_snr": 1.0,
            "architecture": {"family": "classical"}, "gpu_model": None,
            "precision": "float32", "checkpoint_path": None,
            "train_distribution_id": "local", "train_distribution": {},
            "test_distribution": {}, "context_length": 2,
        }
        metadata = {"protocol": {"linear_probe": False}}
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.json"
            aware = Path(directory) / "aware.json"
            legacy.write_text(json.dumps({"schema_version": 1, "metadata": metadata,
                                          "records": [legacy_record]}))
            aware.write_text(json.dumps({"schema_version": 1,
                                         "metadata": {**metadata, "distribution_metadata_version": 1},
                                         "records": [aware_record]}))
            with self.assertRaisesRegex(ValueError, "Cannot combine legacy"):
                load_groups([legacy, aware])

    def test_training_runs_are_equal_weight_after_eval_seed_averaging(self):
        groups = [
            _group("checkpoint-a", 11, [0.0, 0.2], [0.8, 1.0]),
            _group("checkpoint-b", 12, [0.4], [0.6]),
        ]
        summary = summarize_training_replicates(groups, target=0.5)[0]
        cells = {(cell["snr"], cell["k"]): cell for cell in summary["cells"]}
        self.assertAlmostEqual(cells[(1.0, 8)]["metrics"]["bo_candidate"]["mean"], 0.25)
        self.assertEqual(cells[(1.0, 8)]["metrics"]["bo_candidate"]["n_training_runs"], 2)
        self.assertEqual(summary["critical_snr_observations"][0]["status"], "observed_crossings")

    def test_duplicate_or_missing_training_seed_is_not_silently_pooled(self):
        duplicate = [_group("a", 5, [0.1], [0.9]), _group("b", 5, [0.2], [0.8])]
        with self.assertRaisesRegex(ValueError, "Duplicate training seed"):
            summarize_training_replicates(duplicate, 0.5)
        missing = [_group("a", None, [0.1], [0.9]), _group("b", None, [0.2], [0.8])]
        result = summarize_training_replicates(missing, 0.5)[0]
        self.assertEqual(result["status"], "not_aggregated_missing_training_seed_metadata")


if __name__ == "__main__":
    unittest.main()
