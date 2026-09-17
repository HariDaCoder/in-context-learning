"""Tests for the environment-independent batch launcher."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_bo_suite import (
    REPOSITORY_ROOT,
    build_parser,
    checkpoint_dirs,
    evaluation_configs,
    training_configs,
)


class RunBOSuiteTests(unittest.TestCase):
    def test_matrix_commands_use_neutral_group_names(self):
        parser = build_parser()
        planned = parser.parse_args(["plan", "--group", "canonical"])
        trained = parser.parse_args([
            "train-matrix", "--group", "stage0", "--max-concurrent", "2",
            "--dry-run",
        ])
        evaluated = parser.parse_args([
            "evaluate-matrix", "--group", "architecture", "--protocol", "matched",
            "--boundary-suggestions", "results/boundary_suggestions.json", "--dry-run",
        ])
        self.assertEqual(planned.group, "canonical")
        self.assertEqual(trained.max_concurrent, 2)
        self.assertEqual(evaluated.group, ["architecture"])
        self.assertEqual(
            evaluated.boundary_suggestions,
            [Path("results/boundary_suggestions.json")],
        )

    def test_parameter_manifest_is_accepted_by_training_parser(self):
        args = build_parser().parse_args([
            "train-matrix", "--matrix-manifest", "results/parameter_match.json",
            "--dry-run",
        ])
        self.assertIsNone(args.group)
        self.assertEqual(args.matrix_manifest, Path("results/parameter_match.json"))

    def test_all_training_preset_has_six_unique_configs(self):
        configs = training_configs("all")
        self.assertEqual(len(configs), 6)
        self.assertEqual(len(set(configs)), 6)
        self.assertTrue(all(config.is_file() for config in configs))

    def test_boundary_is_explicit(self):
        coarse = evaluation_configs("all")
        with_boundary = evaluation_configs("all", include_boundary=True)
        self.assertEqual(len(coarse), 4)
        self.assertEqual(len(with_boundary), 5)
        self.assertEqual(with_boundary[-1].name, "bo_sweep_boundary.yaml")

    def test_none_preset_runs_only_explicit_configs(self):
        custom = "src/conf/bo_markov.yaml"
        configs = training_configs("none", [custom])
        self.assertEqual([config.name for config in configs], ["bo_markov.yaml"])

    def test_checkpoint_manifest_paths_are_relative_to_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"checkpoint_dirs": ["models/a/run"]}))
            paths = checkpoint_dirs(manifests=[manifest])
        self.assertEqual(paths, ((REPOSITORY_ROOT / "models/a/run").resolve(),))

    def test_core_training_context_covers_largest_core_evaluation(self):
        config_root = REPOSITORY_ROOT / "src" / "conf"
        training = yaml.safe_load((config_root / "bo_iid.yaml").read_text())
        phase = yaml.safe_load((config_root / "bo_sweep_phase.yaml").read_text())
        self.assertEqual(training["training"]["max_context"], 80)
        self.assertEqual(training["training"]["curriculum"]["points"]["end"], 81)
        self.assertGreaterEqual(
            training["training"]["max_context"], max(phase["context_lengths"])
        )

    def test_change_point_train_and_evaluation_use_same_max_context(self):
        config_root = REPOSITORY_ROOT / "src" / "conf"
        forward = yaml.safe_load((config_root / "bo_sweep_change_forward.yaml").read_text())
        reverse = yaml.safe_load((config_root / "bo_sweep_change_reverse.yaml").read_text())
        self.assertEqual(forward["context_lengths"], [80])
        self.assertEqual(reverse["context_lengths"], [80])
        self.assertEqual(forward["feature_change_point"], 40)
        self.assertEqual(reverse["feature_change_point"], 41)


if __name__ == "__main__":
    unittest.main()
