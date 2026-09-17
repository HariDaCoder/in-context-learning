"""Tests for matrix evaluation planning without running expensive sweeps."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_matrix import expand_group
from bo_matrix_eval import (
    build_evaluation_jobs,
    evaluation_config,
    load_boundary_suggestions,
    plot_bundles,
    result_bundles,
    run_job,
    run_jobs,
    write_result_manifest,
)


class MatrixEvaluationPlanningTests(unittest.TestCase):
    def test_boundary_artifact_selects_union_of_followup_snrs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary_suggestions.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "groups": [
                    {"suggestions": [
                        {"suggested_snrs": [0.4, 1.6]},
                        {"suggested_snrs": [0.8, 1.6]},
                    ]}
                ],
            }), encoding="utf-8")
            self.assertEqual(load_boundary_suggestions(path), (0.4, 0.8, 1.6))

    def test_one_baseline_job_per_compatible_checkpoint_bundle(self):
        candidates = [spec for spec in expand_group("stage0") if spec.train_rho_x == 0.0][:2]
        jobs = build_evaluation_jobs(
            candidates, protocols=("matched",), test_snrs=(0.4, 2.0),
            eval_seeds=(1001,), k_over_d=(0.5, 1.0),
        )
        self.assertEqual(len(jobs), 3)
        bundle = next(iter(result_bundles(jobs).values()))
        self.assertEqual(sum(job.kind == "baseline" for job in bundle), 1)
        self.assertEqual(sum(job.kind == "checkpoint" for job in bundle), 2)
        self.assertEqual(next(job for job in bundle if job.kind == "checkpoint").config["baselines"], [])
        self.assertEqual(len(next(job for job in bundle if job.kind == "baseline").config["baselines"]), 3)

    def test_stationary_matched_and_shift_grids_are_separate(self):
        spec = next(item for item in expand_group("stage0") if item.train_rho_x == 0.6)
        matched = evaluation_config(
            spec, "matched", test_snrs=(0.2, 2.0), eval_seeds=(1001, 1002),
            k_over_d=(0.5, 1.0), shift_test_rhos=(0.0, 0.6, 0.9),
        )
        shifted = evaluation_config(
            spec, "shift", test_snrs=(0.2, 2.0), eval_seeds=(1001, 1002),
            k_over_d=(0.5, 1.0), shift_test_rhos=(0.0, 0.6, 0.9),
        )
        self.assertEqual(matched["rhos"], [0.6])
        self.assertEqual(shifted["rhos"], [0.0, 0.9])
        self.assertEqual(matched["snrs"], [0.2, 2.0])
        self.assertEqual(matched["eval_seeds"], [1001, 1002])
        self.assertEqual(matched["context_lengths"], [10, 20])

    def test_change_point_matched_schedule_and_stationary_shift(self):
        spec = next(
            item for item in expand_group("canonical")
            if item.regime == "change_forward" and item.train_seed == 0
        )
        matched = evaluation_config(
            spec, "matched", test_snrs=(2.0,), eval_seeds=(1001,),
            k_over_d=(1.0, 4.0), shift_test_rhos=(0.0, 0.9),
        )
        shifted = evaluation_config(
            spec, "shift", test_snrs=(2.0,), eval_seeds=(1001,),
            k_over_d=(1.0, 4.0), shift_test_rhos=(0.0, 0.9),
        )
        self.assertEqual(matched["rhos"], [spec.train_rho_x])
        self.assertEqual(matched["feature_rho_after"], spec.rho_x_after)
        self.assertEqual(matched["feature_change_point"], spec.feature_change_point)
        self.assertEqual(shifted["rhos"], [0.0, 0.9])
        self.assertIsNone(shifted["feature_rho_after"])
        self.assertIsNone(shifted["feature_change_point"])

    def test_shift_grid_that_contains_only_matched_rho_is_empty(self):
        spec = expand_group("stage0")[0]
        config = evaluation_config(
            spec, "shift", test_snrs=(2.0,), eval_seeds=(1001,),
            k_over_d=(1.0,), shift_test_rhos=(spec.train_rho_x,),
        )
        self.assertIsNone(config)


class MatrixEvaluationExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        spec = expand_group("stage0")[0]
        self.jobs = build_evaluation_jobs(
            (spec,), protocols=("matched",), test_snrs=(2.0,),
            eval_seeds=(1001,), k_over_d=(1.0,),
            output_root=Path("results/eval_test"),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_dry_run_materializes_yaml_and_manifest_without_checkpoint(self):
        statuses = run_jobs(self.jobs, self.root, dry_run=True, max_workers=2)
        self.assertTrue(all(row["status"] == "planned" for row in statuses))
        for job in self.jobs:
            self.assertTrue((self.root / job.config_path).is_file())
            self.assertFalse((self.root / job.result_path).exists())
        plots = plot_bundles(
            self.jobs, statuses, self.root, Path("results/eval_test"), dry_run=True,
            summary_only=True,
        )
        self.assertEqual(plots[0]["status"], "planned")
        self.assertEqual(plots[0]["result_count"], len(self.jobs))
        manifest = write_result_manifest(
            self.jobs, statuses, plots, self.root, Path("results/eval_test")
        )
        document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(document["status_counts"], {"planned": len(self.jobs)})

    def test_valid_existing_result_is_skipped_before_checkpoint_check(self):
        checkpoint_job = next(job for job in self.jobs if job.kind == "checkpoint")
        result = self.root / checkpoint_job.result_path
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"schema_version": 1, "records": []}))
        status = run_job(checkpoint_job, self.root)
        self.assertEqual(status["status"], "skipped_existing")

    def test_failed_process_writes_log_and_failure_record(self):
        baseline_job = next(job for job in self.jobs if job.kind == "baseline")
        completed = mock.Mock(returncode=7)
        with mock.patch("bo_matrix_eval.subprocess.run", return_value=completed):
            status = run_job(baseline_job, self.root)
        self.assertEqual(status["status"], "failed")
        self.assertTrue((self.root / baseline_job.log_path).is_file())
        failure = self.root / baseline_job.failure_path
        self.assertTrue(failure.is_file())
        self.assertEqual(json.loads(failure.read_text())["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
