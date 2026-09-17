"""Tests for neutral architecture generation and exact parameter matching."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_architecture import (
    ArchitectureSpec,
    architecture_count_rows,
    architecture_specs,
    fixed_head_dim_specs,
    fixed_width_depth_specs,
    fixed_width_head_specs,
    parameter_matched_depth_specs,
    write_architecture_csv,
)


class FakeParameter:
    def __init__(self, size):
        self.size = size
        self.requires_grad = True

    def numel(self):
        return self.size


class SizedModel:
    def __init__(self, config):
        # Count = width * depth + width + n_dims.  This deliberately differs
        # from a quadratic Transformer formula so matching must instantiate.
        self._parameters = (
            FakeParameter(config.n_layer * config.n_embd),
            FakeParameter(config.n_embd),
            FakeParameter(config.n_dims),
        )

    def parameters(self):
        return iter(self._parameters)


class ArchitectureTests(unittest.TestCase):
    def test_three_families_have_declared_invariants(self):
        fixed_width = fixed_width_head_specs(heads=(1, 2, 4), n_embd=64, n_layer=6)
        self.assertEqual({item.n_embd for item in fixed_width}, {64})
        self.assertEqual([item.n_head for item in fixed_width], [1, 2, 4])

        fixed_dim = fixed_head_dim_specs(heads=(1, 2, 4), head_dim=16, n_layer=6)
        self.assertEqual({item.head_dim for item in fixed_dim}, {16})
        self.assertEqual([item.n_embd for item in fixed_dim], [16, 32, 64])

        depths = fixed_width_depth_specs(depths=(2, 5), n_embd=64, n_head=4)
        self.assertEqual([item.n_layer for item in depths], [2, 5])
        self.assertEqual({(item.n_embd, item.n_head) for item in depths}, {(64, 4)})
        self.assertEqual(len(architecture_specs()), 14)

    def test_parameter_match_uses_instantiated_exact_counts(self):
        target = ArchitectureSpec(24, 4, 4)
        matches = parameter_matched_depth_specs(
            target=target,
            depths=(2, 8),
            candidate_widths=(8, 16, 24, 32, 40),
            n_dims=3,
            n_positions=17,
            n_head=4,
            model_factory=SizedModel,
        )
        target_count = 24 * 4 + 24 + 3
        self.assertTrue(all(match.target_parameter_count == target_count for match in matches))
        for match in matches:
            expected = (
                match.candidate.n_embd * match.candidate.n_layer
                + match.candidate.n_embd + 3
            )
            self.assertEqual(match.parameter_count, expected)
            self.assertEqual(match.absolute_error, abs(expected - target_count))
            self.assertAlmostEqual(match.relative_error, match.absolute_error / target_count)
        self.assertEqual(matches[0].candidate.n_embd, 40)
        self.assertEqual(matches[1].candidate.n_embd, 16)

    def test_architecture_count_report_instantiates_every_spec(self):
        specs = fixed_width_depth_specs(depths=(2, 4), n_embd=16, n_head=4)
        rows = architecture_count_rows(
            specs, n_dims=3, n_positions=10, model_factory=SizedModel
        )
        self.assertEqual([row["parameter_count"] for row in rows], [51, 83])
        self.assertTrue(all(row["n_positions"] == 10 for row in rows))

    def test_csv_reports_exact_counts_and_errors(self):
        match = parameter_matched_depth_specs(
            ArchitectureSpec(16, 4, 4), (2,), (8, 16, 24), 3, 10,
            model_factory=SizedModel,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_architecture_csv(match, Path(directory) / "report.csv")
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertIn("parameter_count", rows[0])
        self.assertIn("relative_parameter_error", rows[0])

    def test_invalid_divisibility_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ArchitectureSpec(30, 4, 8)


if __name__ == "__main__":
    unittest.main()
