import argparse
from collections import defaultdict
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_DATASETS = {
    "node": {
        "Cora-ML": "Cora-ML",
        "CiteSeer": "CiteSeer",
        "PubMed": "PubMed",
        "Amazon-C": "computers",
    },
    "graph": {
        "AIDS": "AIDS",
        "PROTEINS": "PROTEINS",
        "DD": "DD",
        "MUTAG": "MUTAG",
    },
    "real_world": {
        "Amazon2M": "Products",
        "Big-Vul": "Big-Vul",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all paper-overlap checkpoint benchmarks and aggregate results."
    )
    parser.add_argument("--checkpoint-root", default="./checkpoints")
    parser.add_argument("--output-root", default="./paper_overlap_outputs")
    parser.add_argument("--route-confidence", type=float, default=0.85)
    parser.add_argument(
        "--early-stop-ratio",
        type=float,
        default=0.5,
        help="Legacy compatibility argument. Current early stopping ignores this value.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scan_checkpoint_configs(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    configs = []
    for checkpoint_path in checkpoint_root.rglob("best_model.zip"):
        parts = checkpoint_path.parts
        if len(parts) < 6:
            continue
        robust_mode = parts[-5].replace("robust_", "")
        paper = parts[-4]
        dataset = parts[-3]
        try:
            t_value = int(parts[-2])
        except ValueError:
            continue
        configs.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "robust_mode": robust_mode,
                "paper": paper,
                "dataset": dataset,
                "T": t_value,
            }
        )
    return configs


def build_overlap_jobs(configs):
    repo_to_paper = {}
    for task_group, mapping in PAPER_DATASETS.items():
        for paper_name, repo_name in mapping.items():
            repo_to_paper[repo_name] = (task_group, paper_name)

    jobs = []
    for config in configs:
        dataset = config["dataset"]
        if dataset not in repo_to_paper:
            continue
        task_group, paper_name = repo_to_paper[dataset]
        if task_group == "real_world":
            task = "amazon" if dataset == "Products" else None
        else:
            task = task_group
        if task is None:
            continue
        jobs.append(
            {
                **config,
                "task": task,
                "paper_dataset_name": paper_name,
                "repo_dataset_name": dataset,
            }
        )
    jobs.sort(
        key=lambda item: (
            item["task"],
            item["repo_dataset_name"],
            item["robust_mode"],
            item["T"],
        )
    )
    return jobs


def run_benchmark(job, output_root, route_confidence, early_stop_ratio, force=False):
    config_slug = (
        f"{job['task']}_{job['robust_mode']}_{job['repo_dataset_name']}"
        f"_{job['paper']}_T{job['T']}"
    )
    run_dir = Path(output_root) / config_slug
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "benchmark.log"
    if summary_path.exists() and not force:
        with summary_path.open("r", encoding="utf-8") as f:
            return True, json.load(f)

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "benchmark_adaptive_inference.py",
        "--task",
        job["task"],
        "--robust-mode",
        job["robust_mode"],
        "--paper",
        job["paper"],
        "--dataset",
        job["repo_dataset_name"],
        "--T",
        str(job["T"]),
        "--route-confidence",
        str(route_confidence),
        "--early-stop-ratio",
        str(early_stop_ratio),
        "--output-dir",
        str(run_dir),
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with log_path.open("w", encoding="utf-8") as f:
        f.write(completed.stdout or "")

    if completed.returncode != 0:
        return False, {
            "returncode": completed.returncode,
            "log_path": str(log_path),
            "message": (completed.stdout or "").splitlines()[-1] if completed.stdout else "",
        }

    with summary_path.open("r", encoding="utf-8") as f:
        return True, json.load(f)


def flatten_result(job, summary):
    baseline_metrics = summary["baseline"]["metrics"]
    baseline_runtime = summary["baseline"]["runtime"]
    adaptive_metrics = summary["adaptive"]["metrics"]
    adaptive_runtime = summary["adaptive"]["runtime"]
    comparison = summary["comparison"]

    return {
        "paper_dataset_name": job["paper_dataset_name"],
        "repo_dataset_name": job["repo_dataset_name"],
        "task": job["task"],
        "robust_mode": job["robust_mode"],
        "paper": job["paper"],
        "T": job["T"],
        "checkpoint_path": summary["config"]["checkpoint_path"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "adaptive_accuracy": adaptive_metrics["accuracy"],
        "accuracy_delta": comparison["accuracy_delta"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "adaptive_macro_f1": adaptive_metrics["macro_f1"],
        "macro_f1_delta": comparison["macro_f1_delta"],
        "baseline_total_sec": baseline_runtime["total_sec"],
        "adaptive_total_sec": adaptive_runtime["total_sec"],
        "speedup": comparison["speedup"],
        "avg_subgraphs_saved": comparison["avg_subgraphs_saved"],
        "direct_route_ratio": summary["adaptive"]["details"]["direct_route_ratio"],
        "early_stop_ratio_realized": summary["adaptive"]["details"][
            "early_stop_ratio_realized"
        ],
        "base_confidence_mean": summary["adaptive"]["details"]["base_confidence_mean"],
    }


def write_csv(rows, output_path):
    fieldnames = [
        "paper_dataset_name",
        "repo_dataset_name",
        "task",
        "robust_mode",
        "paper",
        "T",
        "checkpoint_path",
        "baseline_accuracy",
        "adaptive_accuracy",
        "accuracy_delta",
        "baseline_macro_f1",
        "adaptive_macro_f1",
        "macro_f1_delta",
        "baseline_total_sec",
        "adaptive_total_sec",
        "speedup",
        "avg_subgraphs_saved",
        "direct_route_ratio",
        "early_stop_ratio_realized",
        "base_confidence_mean",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def summarize_groups(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    summary_rows = []
    metric_names = [
        "speedup",
        "accuracy_delta",
        "macro_f1_delta",
        "direct_route_ratio",
        "early_stop_ratio_realized",
        "avg_subgraphs_saved",
    ]
    for group_key, items in sorted(grouped.items()):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary["runs"] = len(items)
        for metric_name in metric_names:
            summary[f"mean_{metric_name}"] = sum(
                item[metric_name] for item in items
            ) / len(items)
        summary_rows.append(summary)
    return summary_rows


def write_group_csv(rows, keys, output_path):
    summary_rows = summarize_groups(rows, keys)
    fieldnames = list(keys) + [
        "runs",
        "mean_speedup",
        "mean_accuracy_delta",
        "mean_macro_f1_delta",
        "mean_direct_route_ratio",
        "mean_early_stop_ratio_realized",
        "mean_avg_subgraphs_saved",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_speedup(rows, output_path):
    sorted_rows = sorted(rows, key=lambda item: item["speedup"], reverse=True)
    labels = [
        f"{row['repo_dataset_name']}-{row['robust_mode']}-T{row['T']}" for row in sorted_rows
    ]
    values = [row["speedup"] for row in sorted_rows]

    plt.figure(figsize=(16, 7))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=70, ha="right")
    plt.ylabel("Speedup")
    plt.title("Adaptive vs Baseline Speedup Across Paper-Overlap Checkpoints")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_tradeoff(rows, output_path):
    plt.figure(figsize=(10, 6))
    colors = {"node": "#1f77b4", "graph": "#ff7f0e", "amazon": "#2ca02c"}
    for task in sorted(set(row["task"] for row in rows)):
        task_rows = [row for row in rows if row["task"] == task]
        plt.scatter(
            [row["accuracy_delta"] for row in task_rows],
            [row["speedup"] for row in task_rows],
            label=task,
            s=70,
            alpha=0.85,
            color=colors.get(task, "#444444"),
        )
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Accuracy Delta (Adaptive - Baseline)")
    plt.ylabel("Speedup")
    plt.title("Accuracy-Time Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_dataset_summary(rows, output_path):
    dataset_rows = summarize_groups(rows, ["paper_dataset_name"])
    dataset_rows.sort(key=lambda item: item["mean_speedup"], reverse=True)
    labels = [row["paper_dataset_name"] for row in dataset_rows]
    speedups = [row["mean_speedup"] for row in dataset_rows]
    acc_deltas = [row["mean_accuracy_delta"] for row in dataset_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(labels, speedups, color="#1f77b4")
    axes[0].invert_yaxis()
    axes[0].set_title("Mean Speedup by Dataset")
    axes[0].set_xlabel("Speedup")

    axes[1].barh(labels, acc_deltas, color="#ff7f0e")
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Mean Accuracy Delta by Dataset")
    axes[1].set_xlabel("Accuracy Delta")

    fig.suptitle("Paper-overlap Dataset Summary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_analysis(rows, skipped, failures):
    speedup_sorted = sorted(rows, key=lambda item: item["speedup"], reverse=True)
    acc_sorted = sorted(rows, key=lambda item: item["accuracy_delta"], reverse=True)
    macro_sorted = sorted(rows, key=lambda item: item["macro_f1_delta"], reverse=True)
    task_rows = summarize_groups(rows, ["task"])
    robust_rows = summarize_groups(rows, ["robust_mode"])
    task_robust_rows = summarize_groups(rows, ["task", "robust_mode"])
    dataset_rows = summarize_groups(rows, ["paper_dataset_name"])
    fastest_datasets = sorted(
        dataset_rows, key=lambda item: item["mean_speedup"], reverse=True
    )[:3]
    slowest_datasets = sorted(dataset_rows, key=lambda item: item["mean_speedup"])[:3]

    avg_speedup = sum(row["speedup"] for row in rows) / max(len(rows), 1)
    avg_acc_delta = sum(row["accuracy_delta"] for row in rows) / max(len(rows), 1)
    avg_macro_delta = sum(row["macro_f1_delta"] for row in rows) / max(len(rows), 1)
    avg_direct_route = sum(row["direct_route_ratio"] for row in rows) / max(len(rows), 1)
    avg_early_stop = (
        sum(row["early_stop_ratio_realized"] for row in rows) / max(len(rows), 1)
    )

    lines = []
    lines.append("# Paper-overlap benchmark analysis")
    lines.append("")
    lines.append(f"- Total completed runs: {len(rows)}")
    lines.append(f"- Total failed runs: {len(failures)}")
    if skipped:
        lines.append(f"- Paper datasets without matching checkpoints: {', '.join(skipped)}")
    lines.append(f"- Average speedup: {avg_speedup:.4f}x")
    lines.append(f"- Average accuracy delta: {avg_acc_delta:+.6f}")
    lines.append(f"- Average macro-F1 delta: {avg_macro_delta:+.6f}")
    lines.append(f"- Average direct-route ratio: {avg_direct_route:.4f}")
    lines.append(f"- Average realized early-stop ratio: {avg_early_stop:.4f}")
    lines.append("")
    lines.append("## Grouped means")
    lines.append("")
    for row in task_rows:
        lines.append(
            f"- Task `{row['task']}`: speedup {row['mean_speedup']:.4f}x, "
            f"accuracy delta {row['mean_accuracy_delta']:+.6f}, "
            f"macro-F1 delta {row['mean_macro_f1_delta']:+.6f}, "
            f"direct-route ratio {row['mean_direct_route_ratio']:.4f}."
        )
    lines.append("")
    for row in robust_rows:
        lines.append(
            f"- Robust mode `{row['robust_mode']}`: speedup {row['mean_speedup']:.4f}x, "
            f"accuracy delta {row['mean_accuracy_delta']:+.6f}, "
            f"macro-F1 delta {row['mean_macro_f1_delta']:+.6f}, "
            f"direct-route ratio {row['mean_direct_route_ratio']:.4f}."
        )
    lines.append("")
    for row in task_robust_rows:
        lines.append(
            f"- Split `{row['task']}` + `{row['robust_mode']}`: speedup {row['mean_speedup']:.4f}x, "
            f"accuracy delta {row['mean_accuracy_delta']:+.6f}, "
            f"macro-F1 delta {row['mean_macro_f1_delta']:+.6f}, "
            f"direct-route ratio {row['mean_direct_route_ratio']:.4f}."
        )
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    if speedup_sorted:
        top = speedup_sorted[0]
        lines.append(
            f"- Fastest gain: {top['repo_dataset_name']} {top['robust_mode']} T={top['T']} "
            f"with {top['speedup']:.4f}x speedup and accuracy delta {top['accuracy_delta']:+.6f}."
        )
    if acc_sorted:
        top = acc_sorted[0]
        lines.append(
            f"- Best accuracy gain: {top['repo_dataset_name']} {top['robust_mode']} T={top['T']} "
            f"with accuracy delta {top['accuracy_delta']:+.6f} and speedup {top['speedup']:.4f}x."
        )
    if macro_sorted:
        top = macro_sorted[0]
        lines.append(
            f"- Best macro-F1 gain: {top['repo_dataset_name']} {top['robust_mode']} T={top['T']} "
            f"with macro-F1 delta {top['macro_f1_delta']:+.6f}."
        )
    for row in fastest_datasets:
        lines.append(
            f"- Fast dataset mean: {row['paper_dataset_name']} averages {row['mean_speedup']:.4f}x "
            f"speedup and {row['mean_accuracy_delta']:+.6f} accuracy delta."
        )
    for row in slowest_datasets:
        lines.append(
            f"- Slow dataset mean: {row['paper_dataset_name']} averages {row['mean_speedup']:.4f}x "
            f"speedup and {row['mean_accuracy_delta']:+.6f} accuracy delta."
        )
    low_route = [row for row in rows if row["direct_route_ratio"] == 0.0]
    if low_route:
        lines.append(
            "- Most runs still rely mainly on subgraph voting; with route_confidence=0.85, "
            "direct original-graph routing is often too strict for these checkpoints."
        )
    if avg_early_stop > 0:
        lines.append(
            "- The main runtime win comes from early stopping inside subgraph voting rather than direct routing."
        )
    if failures:
        lines.append("")
        lines.append("## Remaining failures")
        lines.append("")
        lines.append(
            "- Some checkpoints are present on disk but cannot be reproduced with the current dataset build; "
            "the most common issue is checkpoint-model shape mismatch."
        )
        for failure in failures:
            lines.append(
                f"- {failure['repo_dataset_name']} {failure['robust_mode']} T={failure['T']}: "
                f"{failure['message']} (log: {failure['log_path']})"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_root = (script_dir / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configs = scan_checkpoint_configs(script_dir / args.checkpoint_root)
    jobs = build_overlap_jobs(configs)

    available_paper_names = {job["paper_dataset_name"] for job in jobs}
    all_paper_names = set(PAPER_DATASETS["node"]) | set(PAPER_DATASETS["graph"]) | set(
        PAPER_DATASETS["real_world"]
    )
    skipped = sorted(all_paper_names - available_paper_names)

    summaries = []
    rows = []
    failures = []
    for idx, job in enumerate(jobs, start=1):
        print(
            f"[{idx}/{len(jobs)}] Running {job['task']} {job['repo_dataset_name']} "
            f"{job['robust_mode']} T={job['T']}"
        )
        success, payload = run_benchmark(
            job,
            output_root=output_root,
            route_confidence=args.route_confidence,
            early_stop_ratio=args.early_stop_ratio,
            force=args.force,
        )
        if success:
            summaries.append({"job": job, "summary": payload})
            rows.append(flatten_result(job, payload))
        else:
            failures.append({**job, **payload})
            print(
                f"  failed: {job['repo_dataset_name']} {job['robust_mode']} T={job['T']} "
                f"-> {payload['message']}"
            )

    aggregate_json = {
        "route_confidence": args.route_confidence,
        "early_stop_ratio": args.early_stop_ratio,
        "completed_runs": len(rows),
        "failed_runs": failures,
        "skipped_paper_datasets": skipped,
        "results": rows,
    }

    write_json(aggregate_json, output_root / "paper_overlap_results.json")
    write_csv(rows, output_root / "paper_overlap_results.csv")
    write_group_csv(rows, ["paper_dataset_name"], output_root / "paper_overlap_summary_by_dataset.csv")
    write_group_csv(rows, ["task"], output_root / "paper_overlap_summary_by_task.csv")
    write_group_csv(
        rows,
        ["task", "robust_mode"],
        output_root / "paper_overlap_summary_by_task_robust.csv",
    )
    if rows:
        plot_speedup(rows, output_root / "paper_overlap_speedup.png")
        plot_tradeoff(rows, output_root / "paper_overlap_tradeoff.png")
        plot_dataset_summary(rows, output_root / "paper_overlap_dataset_summary.png")

    analysis_text = build_analysis(rows, skipped, failures)
    with open(output_root / "paper_overlap_analysis.md", "w", encoding="utf-8") as f:
        f.write(analysis_text)

    print(analysis_text)
    print(f"Saved aggregate outputs to: {output_root}")


if __name__ == "__main__":
    main()
