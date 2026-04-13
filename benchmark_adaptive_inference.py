import argparse
import csv
import json
import os
import time

import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from datasets.dataset_loader import load_graph_data, load_node_data
from gnn import GraphGAT, GraphGCN, GraphGSAGE, NodeGAT, NodeGCN, NodeGSAGE
from inference_utils import EARLY_STOP_RULE
from utils import resolve_checkpoint_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark baseline full-subgraph inference vs adaptive routing inference."
    )
    parser.add_argument("--task", choices=["node", "graph", "amazon"], required=True)
    parser.add_argument("--robust-mode", choices=["n", "e"], required=True)
    parser.add_argument("--paper", choices=["GCN", "GAT", "GSAGE"], default="GCN")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--route-confidence", type=float, default=0.85)
    parser.add_argument(
        "--early-stop-ratio",
        type=float,
        default=0.5,
        help="Legacy compatibility argument. Current early stopping ignores this value.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-root", default="./checkpoints")
    return parser.parse_args()


def get_node_split(dataset):
    if dataset == "PubMed":
        return 2000, 600
    if dataset == "computers":
        return 400, 133
    return 150, 50


def get_graph_split(dataset):
    if dataset == "Mutagenicity":
        return 1000, 400
    return 250, 100


def build_node_model(paper, num_x, num_labels, hidden_size=None):
    if paper == "GCN":
        if hidden_size is None:
            return NodeGCN(num_x, num_labels)
        return NodeGCN(num_x, num_labels, hidden_size=hidden_size)
    if paper == "GAT":
        return NodeGAT(num_x, num_labels)
    if paper == "GSAGE":
        return NodeGSAGE(num_x, num_labels)
    raise ValueError(f"Unsupported paper: {paper}")


def build_graph_model(paper, num_x, num_labels):
    if paper == "GCN":
        return GraphGCN(num_x, num_labels)
    if paper == "GAT":
        return GraphGAT(num_x, num_labels)
    if paper == "GSAGE":
        return GraphGSAGE(num_x, num_labels)
    raise ValueError(f"Unsupported paper: {paper}")


def load_amazon_data():
    from ogb.nodeproppred import PygNodePropPredDataset

    datasets = PygNodePropPredDataset(name="ogbn-products")
    graph = datasets[0]
    x = torch.as_tensor(graph.x)
    edge_index = torch.as_tensor(graph.edge_index)
    labels = torch.as_tensor(graph.y).view(-1)
    num_x = x.shape[1]
    num_labels = 47

    train_idx = []
    valid_idx = []
    test_idx = []
    for c in range(num_labels):
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        train_i = idx[: int(0.3 * len(idx))]
        val_i = idx[int(0.3 * len(idx)) : int(0.5 * len(idx))]
        test_i = idx[int(0.5 * len(idx)) : -1]
        train_idx.extend(train_i.tolist())
        valid_idx.extend(val_i.tolist())
        test_idx.extend(test_i.tolist())
    return x, edge_index, labels, num_x, num_labels, train_idx, valid_idx, test_idx


def build_experiment(args):
    robust_dir = f"robust_{args.robust_mode}"
    if args.robust_mode == "n":
        from node_hash import HashAgent, RobustAmazonNodeClassifier, RobustGraphClassifier, RobustNodeClassifier
    else:
        from edge_hash import HashAgent, RobustAmazonNodeClassifier, RobustGraphClassifier, RobustNodeClassifier

    if args.task == "node":
        num_train, num_val = get_node_split(args.dataset)
        data, num_x, num_labels = load_node_data(
            args.dataset,
            num_train=num_train,
            num_val=num_val,
        )
        model = build_node_model(args.paper, num_x, num_labels)
        classifier = RobustNodeClassifier(
            model,
            HashAgent(h="md5", T=args.T),
            torch.as_tensor(data.edge_index),
            torch.as_tensor(data.x),
            torch.as_tensor(data.y),
            data.train_mask,
            data.val_mask,
            data.test_mask,
            num_labels,
        )
        test_selector = data.test_mask
        labels = torch.as_tensor(data.y)
    elif args.task == "graph":
        num_train, num_val = get_graph_split(args.dataset)
        graphs, num_x, num_labels, masks, labels = load_graph_data(
            args.dataset,
            num_train=num_train,
            num_val=num_val,
        )
        model = build_graph_model(args.paper, num_x, num_labels)
        classifier = RobustGraphClassifier(
            model,
            HashAgent(h="md5", T=args.T),
            graphs,
            labels,
            masks[0],
            masks[1],
            masks[2],
            num_labels,
        )
        test_selector = masks[2]
        labels = torch.as_tensor(labels)
    else:
        x, edge_index, labels, num_x, num_labels, train_idx, valid_idx, test_idx = load_amazon_data()
        hidden_size = 64 if args.paper == "GCN" else None
        model = build_node_model(args.paper, num_x, num_labels, hidden_size=hidden_size)
        classifier = RobustAmazonNodeClassifier(
            model,
            HashAgent(h="md5", T=args.T),
            edge_index,
            x,
            labels,
            train_idx,
            valid_idx,
            test_idx,
            num_labels,
        )
        test_selector = test_idx

    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint_root,
        robust_dir,
        args.paper,
        args.dataset,
        args.T,
    )
    return classifier, labels, test_selector, checkpoint_path


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def to_numpy_labels(tensor_like):
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like.detach().cpu().numpy()
    return np.asarray(tensor_like)


def infer_baseline_avg_subgraphs(classifier, task, test_selector):
    if task == "graph":
        idxs = np.arange(len(classifier.graphs))
        test_ids = idxs[test_selector]
        counts = [len(classifier._get_valid_subgraphs(int(graph_idx))) for graph_idx in test_ids]
        return float(np.mean(counts)) if counts else 0.0
    if task == "amazon":
        return float(len(classifier.subgraphs))
    return float(len(classifier._get_subgraphs()))


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_comparison(path, baseline_metrics, adaptive_metrics):
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    baseline_values = [baseline_metrics[name] for name in metric_names]
    adaptive_values = [adaptive_metrics[name] for name in metric_names]
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, baseline_values, width=width, label="Baseline")
    plt.bar(x + width / 2, adaptive_values, width=width, label="Adaptive")
    plt.xticks(x, metric_names, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_efficiency_comparison(path, baseline_stats, adaptive_stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    time_labels = ["total_sec", "ms_per_sample"]
    baseline_time = [baseline_stats["total_sec"], baseline_stats["ms_per_sample"]]
    adaptive_time = [adaptive_stats["total_sec"], adaptive_stats["ms_per_sample"]]
    x = np.arange(len(time_labels))
    width = 0.35
    axes[0].bar(x - width / 2, baseline_time, width=width, label="Baseline")
    axes[0].bar(x + width / 2, adaptive_time, width=width, label="Adaptive")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(time_labels)
    axes[0].set_title("Time Comparison")
    axes[0].legend()

    axes[1].bar(
        ["Baseline", "Adaptive"],
        [
            baseline_stats["avg_subgraphs_used"],
            adaptive_stats["avg_subgraphs_used"],
        ],
    )
    axes[1].set_title("Average Subgraphs Used")

    ratio_labels = ["direct_route_ratio", "early_stop_ratio"]
    baseline_ratio = [
        baseline_stats["direct_route_ratio"],
        baseline_stats["early_stop_ratio"],
    ]
    adaptive_ratio = [
        adaptive_stats["direct_route_ratio"],
        adaptive_stats["early_stop_ratio"],
    ]
    x = np.arange(len(ratio_labels))
    axes[2].bar(x - width / 2, baseline_ratio, width=width, label="Baseline")
    axes[2].bar(x + width / 2, adaptive_ratio, width=width, label="Adaptive")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ratio_labels, rotation=15)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Routing Ratios")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_routing_profile(path, adaptive_details):
    route_modes = adaptive_details.get("route_modes", [])
    confidences = np.asarray(adaptive_details.get("base_confidences", []), dtype=float)
    direct_mask = np.isin(route_modes, ["direct", "direct_fallback"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    mode_counts = adaptive_details.get("route_mode_counts", {})
    labels = list(mode_counts.keys())
    values = list(mode_counts.values())
    axes[0].bar(labels, values)
    axes[0].set_title("Route Mode Counts")
    axes[0].tick_params(axis="x", rotation=20)

    if confidences.size > 0:
        axes[1].hist(confidences[direct_mask], bins=20, alpha=0.7, label="Direct")
        axes[1].hist(confidences[~direct_mask], bins=20, alpha=0.7, label="Subgraph")
    axes[1].set_title("Base Confidence Distribution")
    axes[1].set_xlabel("Confidence")
    if confidences.size > 0:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    classifier, labels, test_selector, checkpoint_path = build_experiment(args)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Please train or place the model first."
        )
    classifier.load_model(checkpoint_path)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            ".",
            "benchmark_outputs",
            f"{args.task}_{args.robust_mode}_{args.paper}_{args.dataset}_T{args.T}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    y_true = to_numpy_labels(labels[test_selector])

    baseline_start = time.perf_counter()
    baseline_out, baseline_margin, baseline_details = classifier.predict(
        test_selector,
        strategy="baseline",
        return_details=True,
    )
    baseline_time = time.perf_counter() - baseline_start

    adaptive_start = time.perf_counter()
    adaptive_out, adaptive_margin, adaptive_details = classifier.predict(
        test_selector,
        strategy="adaptive",
        route_confidence=args.route_confidence,
        early_stop_ratio=args.early_stop_ratio,
        return_details=True,
    )
    adaptive_time = time.perf_counter() - adaptive_start

    baseline_pred = baseline_out.argmax(dim=1).detach().cpu().numpy()
    adaptive_pred = adaptive_out.argmax(dim=1).detach().cpu().numpy()

    baseline_metrics = compute_metrics(y_true, baseline_pred)
    adaptive_metrics = compute_metrics(y_true, adaptive_pred)

    sample_count = max(int(len(y_true)), 1)
    baseline_stats = {
        "total_sec": float(baseline_time),
        "ms_per_sample": float(baseline_time * 1000.0 / sample_count),
        "avg_subgraphs_used": infer_baseline_avg_subgraphs(
            classifier, args.task, test_selector
        ),
        "direct_route_ratio": 0.0,
        "early_stop_ratio": 0.0,
    }
    adaptive_stats = {
        "total_sec": float(adaptive_time),
        "ms_per_sample": float(adaptive_time * 1000.0 / sample_count),
        "avg_subgraphs_used": float(adaptive_details.get("avg_subgraphs_used", 0.0)),
        "direct_route_ratio": float(adaptive_details.get("direct_route_ratio", 0.0)),
        "early_stop_ratio": float(
            adaptive_details.get("early_stop_ratio_realized", 0.0)
        ),
    }

    summary = {
        "config": {
            "task": args.task,
            "robust_mode": args.robust_mode,
            "paper": args.paper,
            "dataset": args.dataset,
            "T": args.T,
            "route_confidence": args.route_confidence,
            "early_stop_ratio": args.early_stop_ratio,
            "early_stop_rule": EARLY_STOP_RULE,
            "checkpoint_path": checkpoint_path,
        },
        "baseline": {
            "metrics": baseline_metrics,
            "runtime": baseline_stats,
            "details": baseline_details,
        },
        "adaptive": {
            "metrics": adaptive_metrics,
            "runtime": adaptive_stats,
            "details": adaptive_details,
        },
        "comparison": {
            "speedup": float(baseline_time / adaptive_time) if adaptive_time > 0 else None,
            "time_saved_sec": float(baseline_time - adaptive_time),
            "accuracy_delta": float(
                adaptive_metrics["accuracy"] - baseline_metrics["accuracy"]
            ),
            "macro_f1_delta": float(
                adaptive_metrics["macro_f1"] - baseline_metrics["macro_f1"]
            ),
            "avg_subgraphs_saved": float(
                baseline_stats["avg_subgraphs_used"] - adaptive_stats["avg_subgraphs_used"]
            ),
        },
    }

    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    save_json(os.path.join(args.output_dir, "adaptive_details.json"), adaptive_details)

    metric_rows = [
        {"method": "baseline", **baseline_metrics, **baseline_stats},
        {"method": "adaptive", **adaptive_metrics, **adaptive_stats},
    ]
    save_csv(
        os.path.join(args.output_dir, "comparison_metrics.csv"),
        metric_rows,
        [
            "method",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "total_sec",
            "ms_per_sample",
            "avg_subgraphs_used",
            "direct_route_ratio",
            "early_stop_ratio",
        ],
    )

    prediction_rows = []
    route_modes = adaptive_details.get("route_modes", ["baseline_only"] * len(y_true))
    used_subgraphs = adaptive_details.get("used_subgraphs_per_sample", [0] * len(y_true))
    confidences = adaptive_details.get("base_confidences", [0.0] * len(y_true))
    baseline_margin_np = baseline_margin.detach().cpu().numpy()
    adaptive_margin_np = adaptive_margin.detach().cpu().numpy()
    for idx in range(len(y_true)):
        prediction_rows.append(
            {
                "sample_id": idx,
                "y_true": int(y_true[idx]),
                "baseline_pred": int(baseline_pred[idx]),
                "adaptive_pred": int(adaptive_pred[idx]),
                "baseline_correct": int(baseline_pred[idx] == y_true[idx]),
                "adaptive_correct": int(adaptive_pred[idx] == y_true[idx]),
                "baseline_margin": float(baseline_margin_np[idx]),
                "adaptive_margin": float(adaptive_margin_np[idx]),
                "route_mode": str(route_modes[idx]),
                "used_subgraphs": int(used_subgraphs[idx]),
                "base_confidence": float(confidences[idx]),
            }
        )
    save_csv(
        os.path.join(args.output_dir, "prediction_comparison.csv"),
        prediction_rows,
        [
            "sample_id",
            "y_true",
            "baseline_pred",
            "adaptive_pred",
            "baseline_correct",
            "adaptive_correct",
            "baseline_margin",
            "adaptive_margin",
            "route_mode",
            "used_subgraphs",
            "base_confidence",
        ],
    )

    plot_metric_comparison(
        os.path.join(args.output_dir, "metrics_comparison.png"),
        baseline_metrics,
        adaptive_metrics,
    )
    plot_efficiency_comparison(
        os.path.join(args.output_dir, "efficiency_comparison.png"),
        baseline_stats,
        adaptive_stats,
    )
    plot_routing_profile(
        os.path.join(args.output_dir, "routing_profile.png"),
        adaptive_details,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reports saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
