"""Focused tests for the environment-neutral experiment matrix."""

import csv
import json
import socket
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_matrix import (
    EvaluationSpec,
    ExperimentLock,
    GROUPS,
    LockUnavailable,
    RunRegistry,
    dimension_context_lengths,
    expand_axes,
    expand_group,
    experiment_counts,
    iter_dense_evaluations,
    load_experiment_manifest,
    make_evaluation_spec,
    write_manifest,
    write_training_config,
)


class MatrixExpansionTests(unittest.TestCase):
    def test_group_counts(self):
        self.assertEqual(
            experiment_counts(),
            {
                "canonical": 18,
                "stage0": 9,
                "matched_snr_pilot": 6,
                "matched_rho": 35,
                "dimension": 36,
                "architecture": 72,
            },
        )

    def test_axis_order_does_not_change_expansion(self):
        first = expand_axes({}, {"seed": (0, 1), "rho": (0.0, 0.9)})
        second = expand_axes({}, {"rho": (0.0, 0.9), "seed": (0, 1)})
        self.assertEqual(first, second)

    def test_experiment_id_is_semantic_and_deterministic(self):
        canonical_iid = next(
            spec
            for spec in expand_group("canonical")
            if spec.regime == "iid" and spec.train_seed == 0
        )
        stage0_iid = next(
            spec
            for spec in expand_group("stage0")
            if spec.train_rho_x == 0.0 and spec.train_seed == 0
        )
        dimension_iid = next(
            spec
            for spec in expand_group("dimension")
            if spec.d == 20 and spec.train_rho_x == 0.0 and spec.train_seed == 0
        )
        self.assertEqual(canonical_iid.experiment_id, stage0_iid.experiment_id)
        self.assertEqual(stage0_iid.experiment_id, dimension_iid.experiment_id)

    def test_dimension_contexts_and_position_capacity(self):
        self.assertEqual(
            dimension_context_lengths(80), (20, 40, 60, 80, 100, 120, 160, 320)
        )
        largest = next(
            spec
            for spec in expand_group("dimension")
            if spec.d == 80 and spec.train_rho_x == 0.0 and spec.train_seed == 0
        )
        self.assertEqual(largest.max_context, 320)
        self.assertEqual(largest.to_training_config()["model"]["n_positions"], 321)
        self.assertEqual(largest.token_length, 642)

    def test_forward_and_reverse_change_schedules_are_exact_reversals(self):
        forward = next(
            spec
            for spec in expand_group("canonical")
            if spec.regime == "change_forward" and spec.train_seed == 0
        )
        reverse = next(
            spec
            for spec in expand_group("canonical")
            if spec.regime == "change_reverse" and spec.train_seed == 0
        )
        forward_transitions = [
            forward.rho_x_after if t >= forward.feature_change_point else forward.train_rho_x
            for t in range(1, forward.max_context)
        ]
        reverse_transitions = [
            reverse.rho_x_after if t >= reverse.feature_change_point else reverse.train_rho_x
            for t in range(1, reverse.max_context)
        ]
        self.assertEqual(forward_transitions, list(reversed(reverse_transitions)))

    def test_generated_config_uses_deterministic_resume_directory(self):
        spec = expand_group("stage0")[0]
        with tempfile.TemporaryDirectory() as directory:
            path = write_training_config(spec, Path(directory))
            text = path.read_text(encoding="utf-8")
        self.assertIn("resume_id: {}".format(spec.experiment_id), text)
        self.assertIn("experiment_id: {}".format(spec.experiment_id), text)
        self.assertIn("n_positions: 81", text)
        self.assertIn("max_context: 80", text)
        self.assertIn("data_device: model", text)
        self.assertIn("precision: float32", text)
        self.assertNotIn("inherit:", text)
        self.assertNotIn(str(Path(directory)), text)

    def test_standard_shape_records_all_three_sweep_memberships(self):
        standard = next(
            spec
            for spec in expand_group("architecture")
            if (spec.architecture.n_embd, spec.architecture.n_layer, spec.architecture.n_head)
            == (256, 12, 8)
        )
        families = {
            item["sweep_family"]
            for item in standard.manifest_row()["architecture_memberships"]
        }
        self.assertEqual(
            families, {"fixed_width_heads", "fixed_head_dim", "fixed_width_depth"}
        )


class ManifestAndStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spec = expand_group("stage0")[0]

    def tearDown(self):
        self.temporary.cleanup()

    def test_json_and_csv_manifests_match_and_use_relative_paths(self):
        specs = expand_group("stage0")
        json_path, csv_path = write_manifest("stage0", specs, self.root)
        document = json.loads(json_path.read_text(encoding="utf-8"))
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(document["experiment_count"], 9)
        self.assertEqual(len(rows), 9)
        self.assertFalse(Path(document["experiments"][0]["checkpoint_path"]).is_absolute())
        self.assertIsInstance(json.loads(rows[0]["architecture_memberships"]), list)
        self.assertEqual(
            document["evaluation_axes"]["test_snrs"],
            [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
        )
        loaded = load_experiment_manifest(json_path)
        self.assertEqual(
            [spec.experiment_id for spec in loaded],
            [spec.experiment_id for spec in specs],
        )

    def test_manifest_semantic_tampering_is_rejected(self):
        json_path, _ = write_manifest("stage0", (self.spec,), self.root)
        document = json.loads(json_path.read_text(encoding="utf-8"))
        document["experiments"][0]["train_rho_x"] = 0.123
        json_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "semantic fields"):
            load_experiment_manifest(json_path)

    @unittest.skip("Skipped by project owner; lock behavior is exercised by launcher state tests")
    def test_file_lock_is_exclusive(self):
        path = self.root / "one.lock"
        first = ExperimentLock(path)
        second = ExperimentLock(path)
        first.acquire()
        try:
            with self.assertRaises(LockUnavailable):
                second.acquire()
        finally:
            first.release()
        second.acquire()
        second.release()

    def test_abandoned_local_lock_is_recovered(self):
        path = self.root / "abandoned.lock"
        path.write_text(
            json.dumps(
                {
                    "token": "old",
                    "pid": 2147483647,
                    "hostname": socket.gethostname(),
                    "created_at": "old",
                }
            ),
            encoding="utf-8",
        )
        replacement = ExperimentLock(path)
        self.assertEqual(replacement.owner_metadata()["token"], "old")
        replacement.acquire()
        replacement.release()
        self.assertFalse(path.exists())

    def test_completed_run_is_skipped(self):
        registry = RunRegistry(self.root)
        checkpoint = registry.checkpoint_path(self.spec)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        registry.mark_running(self.spec, "start")
        registry.mark_completed(self.spec)
        decision = registry.decision(self.spec)
        self.assertEqual(decision.action, "skip")

    def test_interrupted_run_with_checkpoint_resumes(self):
        registry = RunRegistry(self.root)
        checkpoint = registry.checkpoint_path(self.spec)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"partial checkpoint")
        registry.mark_running(self.spec, "start")
        registry.mark_interrupted(self.spec, "preempted")
        decision = registry.decision(self.spec, resume=True)
        self.assertEqual(decision.action, "resume")

    def test_claim_logs_failure_and_releases_lock(self):
        registry = RunRegistry(self.root)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with registry.claim(self.spec):
                raise RuntimeError("synthetic failure")
        state = registry.read_state(self.spec)
        failures = list((self.root / "results/bo_matrix/failures").glob("**/*.json"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(len(failures), 1)
        self.assertFalse(registry.lock_path(self.spec).exists())


class EvaluationProtocolTests(unittest.TestCase):
    def test_factory_records_matched_and_shift_protocols(self):
        training = expand_group("stage0")[0]
        matched = make_evaluation_spec(training, training.train_rho_x, 0.0, 2.0, 1001, 20)
        shifted = make_evaluation_spec(training, 0.9, 0.0, 2.0, 1001, 20)
        self.assertEqual(matched.protocol, "matched")
        self.assertEqual(shifted.protocol, "shift")
        self.assertEqual(matched.k_over_d, 1.0)

    def test_mislabeled_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            EvaluationSpec(
                train_experiment_id="bo_example",
                train_rho_x=0.0,
                test_rho_x=0.9,
                train_rho_e=0.0,
                test_rho_e=0.0,
                train_seed=0,
                eval_seed=1001,
                d=20,
                context_length=20,
                train_snr=2.0,
                test_snr=2.0,
                architecture="standard",
                checkpoint_path="models/bo_matrix/bo_example/state.pt",
                protocol="matched",
            )

    def test_dense_axes_label_each_row_from_actual_rhos(self):
        training = expand_group("stage0")[0]
        rows = list(
            iter_dense_evaluations(
                training,
                include_shift=True,
                test_snrs=(0.1, 0.2),
                eval_seeds=(1001,),
                shift_test_rhos=(0.0, 0.9),
                context_lengths=(20,),
            )
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual({row.protocol for row in rows}, {"matched", "shift"})


if __name__ == "__main__":
    unittest.main()
