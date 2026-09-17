"""Tests for conservative critical-SNR scaling fits."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_scaling import fit_log_linear


class BOScalingTests(unittest.TestCase):
    def test_recovers_synthetic_log_linear_exponents(self):
        expected = {"alpha": 0.5, "beta": 0.25, "gamma": -0.1, "c": 1.2}
        rows = []
        for effective_rank, heads, layers in (
            (5, 1, 2), (10, 2, 2), (20, 4, 4), (40, 8, 6),
            (12, 16, 12), (30, 2, 12), (8, 8, 4),
        ):
            log_snr = (
                expected["c"] - expected["alpha"] * math.log(effective_rank)
                - expected["beta"] * math.log(heads)
                - expected["gamma"] * math.log(layers)
            )
            rows.append({
                "effective_rank": effective_rank,
                "n_head": heads,
                "n_layer": layers,
                "critical_snr": math.exp(log_snr),
            })
        result = fit_log_linear(rows)
        self.assertEqual(result["status"], "fit")
        self.assertAlmostEqual(result["alpha_effective_rank"], expected["alpha"], places=10)
        self.assertAlmostEqual(result["beta_heads"], expected["beta"], places=10)
        self.assertAlmostEqual(result["gamma_layers"], expected["gamma"], places=10)

    def test_rank_deficient_or_small_input_is_not_forced(self):
        small = fit_log_linear([{
            "effective_rank": 10, "n_head": 8, "n_layer": 12, "critical_snr": 1,
        }])
        self.assertEqual(small["status"], "insufficient_observations")
        repeated = fit_log_linear([
            {"effective_rank": value, "n_head": 8, "n_layer": 12, "critical_snr": 1}
            for value in (5, 10, 20, 40)
        ])
        self.assertEqual(repeated["status"], "rank_deficient_design")


if __name__ == "__main__":
    unittest.main()
