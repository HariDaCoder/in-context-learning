"""Focused tests for parameter-match artifacts and mechanism dry-run plans."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_architecture import ArchitectureSpec
from bo_architecture_runner import (
    mechanism_evaluation,
    parameter_match_report,
    valid_width_grid,
)
from bo_matrix import ExperimentSpec


class FakeParameter:
    requires_grad = True

    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size


class CountingFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, config):
        self.calls.append((config.n_embd, config.n_layer, config.n_head))
        count = config.n_embd * (config.n_layer + 1) + config.n_dims

        class Model:
            def parameters(self_inner):
                return iter((FakeParameter(count),))

        return Model()


class ArchitectureRunnerTests(unittest.TestCase):
    def test_valid_width_grid_filters_divisibility(self):
        self.assertEqual(valid_width_grid(8, 40, 4, 8), (8, 16, 24, 32, 40))
        with self.assertRaisesRegex(ValueError, "no width"):
            valid_width_grid(9, 15, 2, 8)

    def test_dry_run_neither_instantiates_nor_writes(self):
        factory = CountingFactory()
        with tempfile.TemporaryDirectory() as directory:
            result = parameter_match_report(
                directory,
                target=ArchitectureSpec(24, 4, 4),
                depths=(2, 4),
                width_min=8,
                width_max=40,
                width_step=4,
                n_dims=3,
                max_context=8,
                model_factory=factory,
                dry_run=True,
            )
            self.assertFalse(result.json_path.exists())
            self.assertFalse(result.csv_path.exists())
        self.assertEqual(factory.calls, [])
        self.assertEqual(result.plan["search_model_instantiations"], 19)
        self.assertEqual(result.plan["planned_training_experiment_count"], 12)

    def test_report_has_exact_counts_and_stable_reusable_experiments(self):
        factory = CountingFactory()
        target = ArchitectureSpec(24, 4, 4)
        with tempfile.TemporaryDirectory() as directory:
            first = parameter_match_report(
                directory,
                target=target,
                depths=(2, 4),
                width_min=8,
                width_max=40,
                width_step=4,
                n_dims=3,
                max_context=8,
                model_factory=factory,
            )
            with first.json_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            with first.csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            second = parameter_match_report(
                directory,
                target=target,
                depths=(2, 4),
                width_min=8,
                width_max=40,
                width_step=4,
                n_dims=3,
                max_context=8,
                model_factory=factory,
            )
        self.assertEqual(len(first.matches), 2)
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(document["training_experiment_count"], 12)
        self.assertEqual(len(first.experiments), 12)  # 2 depths x 2 rho x 3 seed
        self.assertEqual(
            [spec.experiment_id for spec in first.experiments],
            [spec.experiment_id for spec in second.experiments],
        )
        self.assertEqual(first.plan["report_id"], second.plan["report_id"])
        self.assertTrue(all(int(row["parameter_count"]) > 0 for row in csv_rows))
        self.assertTrue(all(spec.group == "parameter_matched_depth" for spec in first.experiments))
        example = first.experiments[0]
        reused = ExperimentSpec(
            group="another_matrix_membership",
            regime="another_readable_label",
            train_seed=example.train_seed,
            d=example.d,
            train_rho_x=example.train_rho_x,
            train_rho_e=example.train_rho_e,
            architecture=example.architecture,
            max_context=example.max_context,
            train_snr=example.train_snr,
            batch_size=example.batch_size,
            training_steps=example.training_steps,
            data_device=example.data_device,
            precision=example.precision,
        )
        self.assertEqual(reused.experiment_id, example.experiment_id)
        self.assertEqual(reused.checkpoint_dir, example.checkpoint_dir)

    def test_mechanism_dry_run_does_not_load_checkpoint(self):
        calls = []

        def forbidden_loader(*args):
            calls.append(args)
            raise AssertionError("dry-run must not load")

        result = mechanism_evaluation(
            checkpoint_dir="missing-checkpoint",
            output_csv="missing-output.csv",
            rhos=(0.0, 0.9),
            snrs=(0.8,),
            context_lengths=(20, 40),
            eval_seeds=(1001, 1002),
            n_eval=4,
            batch_size=2,
            dry_run=True,
            checkpoint_loader=forbidden_loader,
        )
        self.assertEqual(calls, [])
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["protocol"]["condition_count"], 8)
        self.assertIn("dummy label", result["protocol"]["query_label"])
        self.assertIn("aggregate", result["protocol"]["retention"])


if __name__ == "__main__":
    unittest.main()
