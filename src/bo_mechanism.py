"""Online, evaluation-only attention diagnostics for the GPT-2 predictor.

Only scalar moments and small head-by-head Gram matrices are retained.  Raw
attention maps and activations are never stored or written.  By default the
statistics describe the final x token, which is the token used to predict an
appended query in the fixed-context evaluation protocol.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


METRIC_DEFINITIONS = {
    "attention_entropy": "Shannon entropy of the causal attention row (nats)",
    "effective_attended_positions": "exp(attention_entropy)",
    "source_center_of_mass": "attention-weighted source token index",
    "source_center_of_mass_fraction": "source center of mass divided by query token index",
    "temporal_lag": "query token index minus source center of mass",
    "attention_temporal_lag_correlation": (
        "Pearson correlation between attention mass and q-source lag over causal sources; "
        "defined as zero when either vector has zero variance"
    ),
    "self_mass": "attention mass on the query token itself",
    "previous_token_mass": "attention mass on the immediately preceding token",
    "x_token_mass": "attention mass on even source indices (x tokens)",
    "y_token_mass": "attention mass on odd source indices (y tokens)",
    "attention_cosine_similarity": "cosine similarity between aggregated attention maps of two heads",
    "head_output_cosine_similarity": "cosine similarity between pre-projection outputs of two heads",
    "head_output_gram_effective_rank": "tr(G)^2 / tr(G^2) for the aggregated head-output Gram matrix",
}


class OnlineMoments:
    """Numerically simple scalar moments; inputs are detached immediately."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0

    def update(self, values):
        values = torch.as_tensor(values).detach().double().reshape(-1)
        values = values[torch.isfinite(values)]
        if not values.numel():
            return
        self.count += int(values.numel())
        self.total += float(values.sum().cpu())
        self.total_square += float(values.square().sum().cpu())

    @property
    def mean(self):
        return self.total / self.count if self.count else None

    @property
    def std(self):
        if self.count < 2:
            return None
        variance = (self.total_square - self.total * self.total / self.count) / (self.count - 1)
        return math.sqrt(max(variance, 0.0))


def _selected_query_indices(sequence_length, mode, device):
    if mode == "last_x":
        last = sequence_length - 1
        if last % 2:
            last -= 1
        return torch.tensor([last], device=device, dtype=torch.long)
    if mode == "all_x":
        return torch.arange(0, sequence_length, 2, device=device)
    if mode == "all_y":
        return torch.arange(1, sequence_length, 2, device=device)
    if mode == "all":
        return torch.arange(sequence_length, device=device)
    raise ValueError("query_positions must be one of: last_x, all_x, all_y, all")


class AttentionMechanismAnalyzer:
    """Aggregate GPT-2 attention and per-head output statistics online."""

    def __init__(self, model, query_positions="last_x"):
        if query_positions not in {"last_x", "all_x", "all_y", "all"}:
            raise ValueError("query_positions must be one of: last_x, all_x, all_y, all")
        self.model = model
        self.query_positions = query_positions
        self._moments = defaultdict(OnlineMoments)
        self._attention_grams = {}
        self._output_grams = {}
        self._attention_entries = defaultdict(int)
        self._output_entries = defaultdict(int)
        self.n_batches = 0
        self.n_examples = 0

    def _core_model(self):
        core = self.model.module if hasattr(self.model, "module") else self.model
        required = ("_combine", "_read_in", "_backbone")
        if any(not hasattr(core, name) for name in required):
            raise TypeError("model must be this project's TransformerModel wrapper")
        if not hasattr(core._backbone, "h"):
            raise TypeError("model backbone must expose GPT-2 blocks as .h")
        return core

    def _update_attention_layer(self, layer, attention, query_indices):
        if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
            raise ValueError("attention tensors must have shape (B,H,T,T)")
        selected = attention.index_select(2, query_indices).detach()
        batch_size, n_heads, n_queries, sequence_length = selected.shape
        # Some attention implementations use a large negative mask before the
        # softmax, while others explicitly zero the upper triangle.  Enforce
        # the causal source set here so future positions can never affect the
        # diagnostics even if a custom implementation returns small residues.
        source_index = torch.arange(sequence_length, device=selected.device)
        causal = source_index.view(1, 1, 1, -1) <= query_indices.view(1, 1, -1, 1)
        selected = selected * causal.to(dtype=selected.dtype)
        normalizer = selected.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(selected.dtype).tiny
        )
        selected = selected / normalizer
        source = torch.arange(sequence_length, device=selected.device, dtype=selected.dtype)
        query = query_indices.to(dtype=selected.dtype).view(1, 1, n_queries)
        entropy = -(selected * selected.clamp_min(torch.finfo(selected.dtype).tiny).log()).sum(-1)
        center = (selected * source).sum(-1)
        lag = query - center
        center_fraction = torch.where(query > 0, center / query.clamp_min(1), torch.ones_like(center))
        gather_index = query_indices.view(1, 1, n_queries, 1).expand(batch_size, n_heads, -1, 1)
        self_mass = selected.gather(-1, gather_index).squeeze(-1)
        previous_index = (query_indices - 1).clamp_min(0)
        previous_index = previous_index.view(1, 1, n_queries, 1).expand(batch_size, n_heads, -1, 1)
        previous_mass = selected.gather(-1, previous_index).squeeze(-1)
        previous_mass = torch.where(
            query_indices.view(1, 1, n_queries) > 0,
            previous_mass,
            torch.zeros_like(previous_mass),
        )
        x_mass = selected[..., 0::2].sum(-1)
        y_mass = selected[..., 1::2].sum(-1)

        # Pearson correlation is computed per task/head/query over only the
        # causal source tokens.  Use at least float32 for stable moments when
        # evaluation itself uses fp16/bfloat16.  A uniform attention row has
        # zero mass variance, so its correlation is explicitly defined as 0.
        correlation_attention = selected.float() if selected.dtype in {
            torch.float16, torch.bfloat16
        } else selected
        correlation_mask = causal.to(dtype=correlation_attention.dtype)
        correlation_count = correlation_mask.sum(-1)
        correlation_lag = (
            query_indices.view(1, 1, n_queries, 1)
            - source_index.view(1, 1, 1, sequence_length)
        ).to(dtype=correlation_attention.dtype)
        mass_mean = (correlation_attention * correlation_mask).sum(-1) / correlation_count
        lag_mean = (correlation_lag * correlation_mask).sum(-1) / correlation_count
        centered_mass = (correlation_attention - mass_mean.unsqueeze(-1)) * correlation_mask
        centered_lag = (correlation_lag - lag_mean.unsqueeze(-1)) * correlation_mask
        mass_ss = centered_mass.square().sum(-1)
        lag_ss = centered_lag.square().sum(-1)
        numerator = (centered_mass * centered_lag).sum(-1)
        denominator = torch.sqrt(mass_ss * lag_ss)
        variance_floor = torch.finfo(correlation_attention.dtype).eps * correlation_count
        correlation = torch.where(
            (mass_ss > variance_floor) & (lag_ss > variance_floor),
            numerator / denominator.clamp_min(torch.finfo(correlation_attention.dtype).tiny),
            torch.zeros_like(numerator),
        )
        metrics = {
            "attention_entropy": entropy,
            "effective_attended_positions": entropy.exp(),
            "source_center_of_mass": center,
            "source_center_of_mass_fraction": center_fraction,
            "temporal_lag": lag,
            "attention_temporal_lag_correlation": correlation,
            "self_mass": self_mass,
            "previous_token_mass": previous_mass,
            "x_token_mass": x_mass,
            "y_token_mass": y_mass,
        }
        for metric, values in metrics.items():
            for head in range(n_heads):
                self._moments[(layer, head, metric)].update(values[:, head])

        vectors = selected.permute(1, 0, 2, 3).reshape(n_heads, -1).double()
        gram = (vectors @ vectors.transpose(0, 1)).cpu()
        if layer not in self._attention_grams:
            self._attention_grams[layer] = torch.zeros_like(gram)
        self._attention_grams[layer] += gram
        self._attention_entries[layer] += batch_size * n_queries * sequence_length

    @staticmethod
    def _head_output_gram(merged_output, query_indices, n_heads):
        if merged_output.ndim != 3 or merged_output.shape[-1] % n_heads:
            raise ValueError("pre-projection head output has incompatible shape")
        selected = merged_output.index_select(1, query_indices).detach()
        batch_size, n_queries, width = selected.shape
        head_dim = width // n_heads
        vectors = selected.reshape(batch_size, n_queries, n_heads, head_dim)
        vectors = vectors.permute(2, 0, 1, 3).reshape(n_heads, -1).double()
        gram = (vectors @ vectors.transpose(0, 1)).cpu()
        return gram, batch_size * n_queries * head_dim

    def _merge_output_gram(self, layer, gram, entries):
        if layer not in self._output_grams:
            self._output_grams[layer] = torch.zeros_like(gram)
        self._output_grams[layer] += gram
        self._output_entries[layer] += entries

    @torch.no_grad()
    def update(self, xs, ys):
        """Run one eval batch and merge its statistics; model mode is restored."""

        if xs.ndim != 3 or ys.shape != xs.shape[:2]:
            raise ValueError("xs and ys must have shapes (B,K,d) and (B,K)")
        if not xs.is_floating_point() or not ys.is_floating_point():
            raise ValueError("xs and ys must be floating point tensors")
        core = self._core_model()
        parameter = next(iter(core.parameters()), None)
        device = parameter.device if parameter is not None else xs.device
        dtype = parameter.dtype if parameter is not None else xs.dtype
        combined = core._combine(xs.to(device=device, dtype=dtype), ys.to(device=device, dtype=dtype))
        query_indices = _selected_query_indices(combined.shape[1], self.query_positions, device)
        # Hooks reduce each layer immediately to an H-by-H CPU Gram matrix.
        # They never retain the B-by-T-by-D activation beyond the hook call.
        captured_output_grams = {}
        handles = []

        def capture(layer, n_heads):
            def hook(_module, inputs):
                if not inputs:
                    raise RuntimeError("GPT-2 projection hook received no input")
                captured_output_grams[layer] = self._head_output_gram(
                    inputs[0], query_indices, n_heads
                )

            return hook

        for layer, block in enumerate(core._backbone.h):
            if not hasattr(block, "attn") or not hasattr(block.attn, "c_proj"):
                raise TypeError("GPT-2 attention block must expose attn.c_proj")
            if not hasattr(block.attn, "num_heads"):
                raise TypeError("GPT-2 attention block must expose attn.num_heads")
            handles.append(block.attn.c_proj.register_forward_pre_hook(
                capture(layer, block.attn.num_heads)
            ))

        was_training = self.model.training
        try:
            self.model.eval()
            embeddings = core._read_in(combined)
            outputs = core._backbone(
                inputs_embeds=embeddings,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            attentions = outputs.attentions
            if attentions is None or len(attentions) != len(core._backbone.h):
                raise RuntimeError("backbone did not return one attention tensor per layer")
            for layer, attention in enumerate(attentions):
                self._update_attention_layer(layer, attention, query_indices)
                if layer not in captured_output_grams:
                    raise RuntimeError("head-output hook was not called for layer {}".format(layer))
                if captured_output_grams[layer][0].shape[0] != attention.shape[1]:
                    raise RuntimeError("attention/head-output head counts disagree at layer {}".format(layer))
                self._merge_output_gram(layer, *captured_output_grams[layer])
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        self.n_batches += 1
        self.n_examples += xs.shape[0]
        return self

    @staticmethod
    def _cosine_rows(layer, gram, metric, entries, query_positions):
        diagonal = gram.diag().clamp_min(0)
        denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :])
        cosine = torch.where(denominator > 0, gram / denominator, torch.zeros_like(gram))
        rows = []
        for head in range(gram.shape[0]):
            for peer in range(head + 1, gram.shape[0]):
                rows.append({
                    "record_type": "head_pair",
                    "layer": layer,
                    "head": head,
                    "peer_head": peer,
                    "metric": metric,
                    "value": float(cosine[head, peer]),
                    "std": None,
                    "n": entries,
                    "query_positions": query_positions,
                    "definition": METRIC_DEFINITIONS[metric],
                })
        return rows

    def rows(self):
        """Return scalar summaries suitable for CSV/analysis dataframes."""

        rows = []
        for (layer, head, metric), moment in sorted(self._moments.items()):
            rows.append({
                "record_type": "head_metric",
                "layer": layer,
                "head": head,
                "peer_head": None,
                "metric": metric,
                "value": moment.mean,
                "std": moment.std,
                "n": moment.count,
                "query_positions": self.query_positions,
                "definition": METRIC_DEFINITIONS[metric],
            })
        for layer, gram in sorted(self._attention_grams.items()):
            rows.extend(self._cosine_rows(
                layer, gram, "attention_cosine_similarity",
                self._attention_entries[layer], self.query_positions,
            ))
        for layer, gram in sorted(self._output_grams.items()):
            rows.extend(self._cosine_rows(
                layer, gram, "head_output_cosine_similarity",
                self._output_entries[layer], self.query_positions,
            ))
            trace = float(gram.trace())
            trace_square = float(gram.square().sum())
            effective_rank = trace * trace / trace_square if trace_square > 0 else 0.0
            rows.append({
                "record_type": "layer_metric",
                "layer": layer,
                "head": None,
                "peer_head": None,
                "metric": "head_output_gram_effective_rank",
                "value": effective_rank,
                "std": None,
                "n": self._output_entries[layer],
                "query_positions": self.query_positions,
                "definition": METRIC_DEFINITIONS["head_output_gram_effective_rank"],
            })
        return rows

    def write_csv(self, path, metadata=None):
        """Write summaries plus constant run metadata; never writes raw tensors."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        fields = (
            "record_type", "layer", "head", "peer_head", "metric", "value",
            "std", "n", "query_positions", "n_batches", "n_examples",
            "definition", "metadata_json",
        )
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.rows():
                writer.writerow({
                    **row,
                    "n_batches": self.n_batches,
                    "n_examples": self.n_examples,
                    "metadata_json": metadata_json,
                })
        return destination
