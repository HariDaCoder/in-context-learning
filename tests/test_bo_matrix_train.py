"""Tests for matrix planning, execution state, and short-run benchmarks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_matrix import expand_group
from bo_matrix_train import (
    benchmark_group,
    plan_group,
    plan_manifest,
    recommended_concurrency,
    train_group,
    train_manifest,
)


class FakeTrainingRunner:
    def __init__(self, memory_bytes=1234, delay=0.0, fail=False):
        self.memory_bytes = memory_bytes
        self.delay = delay
        self.fail = fail
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, command, cwd, stdout, stderr, check):
        with self.lock:
            self.calls.append(tuple(command))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise subprocess.CalledProcessError(7, command)
            config_path = (Path(cwd) / command[-1]).resolve()
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            output_dir = (Path(cwd) / config["out_dir"] / config["training"]["resume_id"]).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "state.pt").write_bytes(b"state")
            (output_dir / "final.pt").write_bytes(b"final")
            completed = {
                "experiment_id": config["training"]["experiment_id"],
                "steps_completed_this_invocation": config["training"]["train_steps"],
                "wall_time_seconds": max(self.delay, 0.001),
                "max_cuda_memory_bytes": self.memory_bytes,
            }
            (output_dir / "completed.json").write_text(
                json.dumps(completed), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0)
        finally:
            with self.lock:
                self.active -= 1


class MatrixTrainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_plan_uses_interpreter_device_and_precision_specific_ids(self):
        default_id = expand_group("stage0")[0].experiment_id
        plan = plan_group(
            "stage0",
            device="cuda:3",
            precision="bfloat16",
            max_concurrent=2,
            repository_root=self.root,
            interpreter="/runtime/python",
        )
        self.assertEqual(plan.counts["start"], 9)
        self.assertEqual(plan.precision, "bfloat16")
        self.assertNotEqual(plan.items[0].spec.experiment_id, default_id)
        self.assertEqual(plan.items[0].command[0], "/runtime/python")
        config = yaml.safe_load(plan.items[0].config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["training"]["device"], "cuda:3")
        self.assertEqual(config["training"]["precision"], "bfloat16")
        self.assertTrue(plan.summary_path.is_file())

    def test_dry_run_never_calls_subprocess(self):
        runner = FakeTrainingRunner()
        summary = train_group(
            "stage0",
            dry_run=True,
            repository_root=self.root,
            runner=runner,
        )
        self.assertEqual(len(runner.calls), 0)
        self.assertEqual(summary.counts["planned"], 9)
        self.assertTrue(summary.summary_path.is_file())

    def test_parameter_match_manifest_can_be_planned_and_trained(self):
        spec = expand_group("stage0")[0]
        manifest = self.root / "parameter_match.json"
        manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "training_experiment_count": 1,
                "training_experiments": [spec.manifest_row()],
            }),
            encoding="utf-8",
        )
        plan = plan_manifest(manifest, repository_root=self.root)
        self.assertEqual(plan.counts["start"], 1)
        runner = FakeTrainingRunner()
        summary = train_manifest(
            manifest, dry_run=True, repository_root=self.root, runner=runner
        )
        self.assertEqual(summary.counts["planned"], 1)
        self.assertEqual(runner.calls, [])

    def test_success_is_skipped_on_second_launch(self):
        runner = FakeTrainingRunner()
        one = (expand_group("stage0")[0],)
        with patch("bo_matrix_train.expand_group", return_value=one):
            first = train_group("stage0", repository_root=self.root, runner=runner)
            second = train_group("stage0", repository_root=self.root, runner=runner)
        self.assertEqual(first.counts["completed"], 1)
        self.assertEqual(second.counts["skipped"], 1)
        self.assertEqual(len(runner.calls), 1)

    def test_thread_pool_respects_requested_parallelism(self):
        runner = FakeTrainingRunner(delay=0.05)
        two = expand_group("stage0")[:2]
        with patch("bo_matrix_train.expand_group", return_value=two):
            summary = train_group(
                "stage0",
                max_concurrent=2,
                repository_root=self.root,
                runner=runner,
            )
        self.assertEqual(summary.counts["completed"], 2)
        self.assertEqual(runner.max_active, 2)

    def test_failure_has_log_registry_state_and_summary(self):
        runner = FakeTrainingRunner(fail=True)
        one = (expand_group("stage0")[0],)
        with patch("bo_matrix_train.expand_group", return_value=one):
            summary = train_group("stage0", repository_root=self.root, runner=runner)
        self.assertEqual(summary.counts["failed"], 1)
        self.assertFalse(summary.success)
        self.assertTrue((self.root / summary.results[0].log_path).is_file())
        state_files = list((self.root / "results/bo_matrix/state").glob("*.json"))
        failure_files = list((self.root / "results/bo_matrix/failures").glob("**/*.json"))
        self.assertEqual(json.loads(state_files[0].read_text())["status"], "failed")
        self.assertEqual(len(failure_files), 1)

    def test_benchmark_uses_isolated_ids_and_reports_throughput(self):
        runner = FakeTrainingRunner(memory_bytes=100, delay=0.01)
        four = expand_group("stage0")[:4]
        with patch("bo_matrix_train.expand_group", return_value=four):
            report = benchmark_group(
                "stage0",
                concurrency_values=(1, 2),
                steps=5,
                repository_root=self.root,
                runner=runner,
                benchmark_run_id="test_benchmark",
            )
        first, second = report["trials"]
        self.assertTrue(first["success"] and second["success"])
        self.assertEqual(first["total_steps_completed"], 5)
        self.assertEqual(second["total_steps_completed"], 10)
        self.assertEqual(second["sum_max_cuda_memory_bytes"], 200)
        self.assertGreater(second["aggregate_steps_per_second"], 0)
        self.assertTrue(all("benchmarks/test_benchmark" in path for path in second["checkpoint_dirs"]))
        self.assertTrue(set(first["experiment_ids"]).isdisjoint(second["experiment_ids"]))
        self.assertTrue((self.root / report["summary_path"]).is_file())
        self.assertEqual(report["recommended_max_concurrent"], 2)
        self.assertEqual(recommended_concurrency("stage0", self.root), 2)


if __name__ == "__main__":
    unittest.main()
