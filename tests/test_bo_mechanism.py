"""Tests for online attention aggregation without raw activation dumps."""

import csv
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bo_mechanism import AttentionMechanismAnalyzer, OnlineMoments


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Same public attribute used by transformers==4.17 GPT2Attention.
        self.num_heads = 2
        self.c_proj = torch.nn.Identity()


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()


class FakeBackbone(torch.nn.Module):
    def __init__(self, width=4, heads=2, layers=2):
        super().__init__()
        self.h = torch.nn.ModuleList([FakeBlock() for _ in range(layers)])
        self.width = width
        self.heads = heads

    def forward(self, inputs_embeds, **_kwargs):
        hidden = inputs_embeds
        attentions = []
        batch, tokens, _ = hidden.shape
        for layer, block in enumerate(self.h):
            # Distinct head slices make a nontrivial but compact output Gram.
            hidden = block.attn.c_proj(hidden + (layer + 1) * 0.1)
            uniform = torch.tril(torch.ones(tokens, tokens, device=hidden.device))
            uniform = uniform / uniform.sum(-1, keepdim=True)
            self_only = torch.eye(tokens, device=hidden.device)
            attention = torch.stack((uniform, self_only), dim=0)
            attentions.append(attention.unsqueeze(0).expand(batch, -1, -1, -1))
        return SimpleNamespace(attentions=tuple(attentions))


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._read_in = torch.nn.Linear(2, 4, bias=False)
        self._backbone = FakeBackbone()

    @staticmethod
    def _combine(xs, ys):
        wide_y = torch.stack((ys, torch.zeros_like(ys)), dim=-1)
        return torch.stack((xs, wide_y), dim=2).reshape(xs.shape[0], 2 * xs.shape[1], 2)


class MechanismTests(unittest.TestCase):
    def test_online_moments(self):
        moments = OnlineMoments()
        moments.update(torch.tensor([1.0, 2.0]))
        moments.update(torch.tensor([3.0, float("nan")]))
        self.assertEqual(moments.count, 3)
        self.assertEqual(moments.mean, 2.0)
        self.assertEqual(moments.std, 1.0)

    def test_attention_metrics_and_mode_restoration(self):
        torch.manual_seed(3)
        model = FakeTransformer()
        model.train()
        analyzer = AttentionMechanismAnalyzer(model, query_positions="last_x")
        xs = torch.randn(3, 3, 2)
        ys = torch.randn(3, 3)
        analyzer.update(xs, ys).update(xs, ys)
        self.assertTrue(model.training)
        self.assertEqual(analyzer.n_batches, 2)
        self.assertEqual(analyzer.n_examples, 6)
        rows = analyzer.rows()
        lookup = {(row["layer"], row["head"], row["metric"]): row for row in rows}
        # Final x query is token 4. Uniform causal attention has effective size 5;
        # the self-only head has effective size 1.
        self.assertAlmostEqual(lookup[(0, 0, "effective_attended_positions")]["value"], 5.0, places=5)
        self.assertAlmostEqual(lookup[(0, 1, "effective_attended_positions")]["value"], 1.0, places=5)
        self.assertAlmostEqual(lookup[(0, 0, "x_token_mass")]["value"], 3 / 5, places=5)
        self.assertAlmostEqual(lookup[(0, 0, "y_token_mass")]["value"], 2 / 5, places=5)
        # Uniform mass has zero variance and follows the documented safe value;
        # a self-only head puts more mass at short lag and is negatively related.
        self.assertEqual(
            lookup[(0, 0, "attention_temporal_lag_correlation")]["value"], 0.0
        )
        self.assertLess(
            lookup[(0, 1, "attention_temporal_lag_correlation")]["value"], 0.0
        )
        ranks = [row for row in rows if row["metric"] == "head_output_gram_effective_rank"]
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(1 <= row["value"] <= 2 for row in ranks))
        self.assertTrue(any(row["metric"] == "attention_cosine_similarity" for row in rows))

    def test_csv_contains_only_aggregates(self):
        model = FakeTransformer()
        analyzer = AttentionMechanismAnalyzer(model)
        analyzer.update(torch.randn(2, 2, 2), torch.randn(2, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = analyzer.write_csv(
                Path(directory) / "mechanism.csv", metadata={"checkpoint": "example"}
            )
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = reader.fieldnames
        self.assertTrue(rows)
        self.assertIn("value", fields)
        self.assertIn("metadata_json", fields)
        self.assertNotIn("attention", fields)
        self.assertNotIn("activation", fields)
        self.assertTrue(all(row["n_batches"] == "1" for row in rows))

    def test_lag_correlation_ignores_noncausal_mass(self):
        analyzer = AttentionMechanismAnalyzer(FakeTransformer())
        # Query token 2 may only use sources 0, 1, 2.  Deliberately place mass
        # on future source 3; masking must make this equivalent to self-only.
        attention = torch.tensor([[[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 99.0],
            [0.0, 0.0, 0.0, 1.0],
        ]]] )
        analyzer._update_attention_layer(0, attention, torch.tensor([2]))
        rows = analyzer.rows()
        row = next(
            item for item in rows
            if item["metric"] == "attention_temporal_lag_correlation"
        )
        self.assertLess(row["value"], 0.0)

    def test_invalid_wrapper_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "TransformerModel"):
            AttentionMechanismAnalyzer(torch.nn.Linear(2, 2)).update(
                torch.randn(1, 2, 2), torch.randn(1, 2)
            )


if __name__ == "__main__":
    unittest.main()
