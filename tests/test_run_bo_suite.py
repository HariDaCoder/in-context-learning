"""Tests for the environment-independent batch launcher."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_bo_suite import REPOSITORY_ROOT, checkpoint_dirs, evaluation_configs, training_configs


class RunBOSuiteTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
