import numpy as np
import torch

EARLY_STOP_RULE = "runner_up_plus_remaining_lt_leader"


def get_mask_indices(mask):
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return mask.nonzero(as_tuple=False).view(-1).long().cpu()
        return mask.view(-1).long().cpu()
    return torch.as_tensor(mask, dtype=torch.long).view(-1).cpu()


def confidence_from_logits(logits):
    if logits.numel() == 0:
        return (
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
        )
    probs = torch.softmax(logits.detach(), dim=1)
    confidences, predictions = probs.max(dim=1)
    return confidences.cpu(), predictions.cpu()


def should_early_stop_by_remaining_votes(votes, used_subgraphs, total_subgraphs):
    votes_cpu = votes.detach().cpu()
    if votes_cpu.ndim == 1:
        votes_cpu = votes_cpu.unsqueeze(0)
    if votes_cpu.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool)

    used_tensor = torch.as_tensor(used_subgraphs, dtype=torch.long).view(-1).cpu()
    total_tensor = torch.as_tensor(total_subgraphs, dtype=torch.long).view(-1).cpu()
    if used_tensor.numel() == 1 and votes_cpu.shape[0] != 1:
        used_tensor = used_tensor.repeat(votes_cpu.shape[0])
    if total_tensor.numel() == 1 and votes_cpu.shape[0] != 1:
        total_tensor = total_tensor.repeat(votes_cpu.shape[0])
    if used_tensor.shape[0] != votes_cpu.shape[0]:
        raise ValueError("used_subgraphs must align with votes rows")
    if total_tensor.shape[0] != votes_cpu.shape[0]:
        raise ValueError("total_subgraphs must align with votes rows")

    if votes_cpu.shape[1] == 0:
        return torch.zeros(votes_cpu.shape[0], dtype=torch.bool)
    if votes_cpu.shape[1] == 1:
        remaining = torch.clamp(total_tensor - used_tensor, min=0)
        return remaining < votes_cpu[:, 0].to(torch.long)

    top_votes = torch.topk(votes_cpu, k=2, dim=1).values.to(torch.long)
    leader_votes = top_votes[:, 0]
    runner_up_votes = top_votes[:, 1]
    remaining = torch.clamp(total_tensor - used_tensor, min=0)
    return (runner_up_votes + remaining) < leader_votes


def compute_vote_margin(votes):
    votes_cpu = votes.detach().cpu()
    if votes_cpu.ndim == 1:
        votes_cpu = votes_cpu.unsqueeze(0)
    if votes_cpu.shape[0] == 0:
        return torch.zeros(0, dtype=torch.float32)

    working_votes = votes_cpu.clone()
    vote_label = working_votes.argmax(dim=1)
    margin = torch.zeros(working_votes.shape[0], dtype=torch.float32)
    row_idx = torch.arange(working_votes.shape[0])

    working_votes[row_idx, vote_label] = -working_votes[row_idx, vote_label]
    second_label = working_votes.argmax(dim=1)
    working_votes[row_idx, vote_label] = -working_votes[row_idx, vote_label]

    first_votes = working_votes[row_idx, vote_label]
    second_votes = working_votes[row_idx, second_label]
    adjust_mask = vote_label > second_label

    if adjust_mask.any():
        margin[adjust_mask] = torch.div(
            first_votes[adjust_mask] - second_votes[adjust_mask] - 1,
            2,
            rounding_mode="floor",
        )
    if (~adjust_mask).any():
        margin[~adjust_mask] = torch.div(
            first_votes[~adjust_mask] - second_votes[~adjust_mask],
            2,
            rounding_mode="floor",
        )
    return margin


def build_adaptive_details(
    strategy,
    route_confidence,
    early_stop_ratio,
    total_subgraphs,
    base_confidences,
    route_modes,
    used_subgraphs,
    total_subgraphs_per_sample=None,
):
    route_modes = np.asarray(route_modes, dtype=object)
    used_subgraphs = np.asarray(used_subgraphs, dtype=int)
    base_confidences = np.asarray(base_confidences, dtype=float)
    total_samples = int(route_modes.shape[0])
    if total_subgraphs_per_sample is None:
        total_subgraphs_per_sample = np.full(total_samples, int(total_subgraphs), dtype=int)
    else:
        total_subgraphs_per_sample = np.asarray(total_subgraphs_per_sample, dtype=int)
        if total_subgraphs_per_sample.shape[0] != total_samples:
            raise ValueError("total_subgraphs_per_sample must align with route_modes")

    direct_mask = np.isin(route_modes, ["direct", "direct_fallback"])
    subgraph_mask = ~direct_mask
    early_stop_mask = route_modes == "subgraph_early_stop"

    route_mode_counts = {}
    for mode in np.unique(route_modes):
        route_mode_counts[str(mode)] = int((route_modes == mode).sum())

    details = {
        "strategy": strategy,
        "route_confidence": float(route_confidence),
        "early_stop_ratio": float(early_stop_ratio),
        "early_stop_rule": EARLY_STOP_RULE,
        "total_subgraphs_available": int(total_subgraphs),
        "total_samples": total_samples,
        "direct_route_count": int(direct_mask.sum()),
        "direct_route_ratio": float(direct_mask.mean()) if total_samples else 0.0,
        "subgraph_route_count": int(subgraph_mask.sum()),
        "subgraph_route_ratio": float(subgraph_mask.mean()) if total_samples else 0.0,
        "early_stop_count": int(early_stop_mask.sum()),
        "early_stop_ratio_realized": (
            float(early_stop_mask.sum() / max(int(subgraph_mask.sum()), 1))
            if total_samples
            else 0.0
        ),
        "avg_subgraphs_used": float(used_subgraphs.mean()) if total_samples else 0.0,
        "avg_subgraphs_used_subgraph_only": (
            float(used_subgraphs[subgraph_mask].mean()) if subgraph_mask.any() else 0.0
        ),
        "avg_subgraphs_available": (
            float(total_subgraphs_per_sample.mean()) if total_samples else 0.0
        ),
        "avg_subgraphs_saved": (
            float((total_subgraphs_per_sample - used_subgraphs).mean()) if total_samples else 0.0
        ),
        "base_confidence_mean": float(base_confidences.mean()) if total_samples else 0.0,
        "base_confidence_std": float(base_confidences.std()) if total_samples else 0.0,
        "base_confidence_min": float(base_confidences.min()) if total_samples else 0.0,
        "base_confidence_max": float(base_confidences.max()) if total_samples else 0.0,
        "route_mode_counts": route_mode_counts,
        "route_modes": route_modes.tolist(),
        "used_subgraphs_per_sample": used_subgraphs.tolist(),
        "total_subgraphs_per_sample": total_subgraphs_per_sample.tolist(),
        "base_confidences": base_confidences.tolist(),
    }
    return details
