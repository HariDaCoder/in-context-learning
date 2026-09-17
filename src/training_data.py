"""Prompt construction shared by Transformer training protocols."""

import torch


def sample_training_batch(data_sampler, task_sampler, curriculum, bsize, query_mode,
                          data_sampler_args, task_sampler_args):
    """Draw one prompt, with an optional query independent of the context chain.

    In ``independent`` mode, all but the final feature form the requested
    dependent context. The final feature and its training noise are independent
    marginal draws. The query label is never visible to the causal prediction.
    """
    xs = data_sampler.sample_xs(
        curriculum.n_points,
        bsize,
        curriculum.n_dims_truncated,
        **data_sampler_args,
    )
    task = task_sampler(**task_sampler_args)
    if query_mode == "causal":
        return xs, task.evaluate(xs), task
    if query_mode != "independent":
        raise ValueError("query_mode must be 'causal' or 'independent'")

    query = torch.randn(
        bsize, 1, xs.shape[-1], dtype=xs.dtype, device=xs.device
    )
    query[:, :, curriculum.n_dims_truncated:] = 0
    xs[:, -1:] = query
    context_ys = task.evaluate(xs[:, :-1])
    query_ys = task.evaluate_clean(query)
    if task.noise_std:
        query_ys = query_ys + task.noise_std * torch.randn_like(query_ys)
    return xs, torch.cat((context_ys, query_ys), dim=1), task
