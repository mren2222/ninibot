#!/usr/bin/env python3
"""Offline evaluation for a structured imitation policy."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch

from train_structured_policy import StructuredPolicy
from train_structured_policy import discover_episodes
from train_structured_policy import encode_row
from train_structured_policy import target_exceeds_abs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay episode CSV rows through a structured policy."
    )
    parser.add_argument(
        "--model",
        default="data/models/structured_policy_fine_align/policy.pt",
        help="Path to policy.pt.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.json. Defaults to the model directory metadata.",
    )
    parser.add_argument("--episode-root", default="data/episodes")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "train", "all"],
        help="Episodes to evaluate. val uses the training metadata val_episodes.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        default=None,
        help="Evaluate only this episode id. Repeatable. Overrides --split.",
    )
    parser.add_argument("--include-label", action="append", default=["success"])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for metrics.json and predictions.csv.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="CPU is recommended for this tiny model.",
    )
    parser.add_argument(
        "--sign-eps",
        type=float,
        default=1e-6,
        help="Ignore near-zero targets when computing sign agreement.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Print and save the worst rows by total absolute action error.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a target-vs-prediction time-series PNG.",
    )
    return parser.parse_args()


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path):
    if path is None or not path.exists():
        return {}

    with path.open() as stream:
        return json.load(stream)


def resolve_output_dir(args):
    if args.output_dir is not None:
        return Path(args.output_dir)

    return Path(args.model).parent / f"eval_{args.split}"


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")

    return torch.device(name)


def load_model(model_path, device):
    blob = torch_load(model_path)
    model = StructuredPolicy(
        input_dim=int(blob["input_dim"]),
        output_dim=int(blob["output_dim"]),
        hidden_dim=int(blob["hidden_dim"]),
        dropout=0.0,
    )
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def selected_episode_ids(args, metadata):
    if args.episode:
        return set(args.episode)

    all_ids = set(metadata.get("episodes", []))
    val_ids = set(metadata.get("val_episodes", []))

    if args.split == "val":
        return val_ids

    if args.split == "train":
        return all_ids - val_ids

    return all_ids


def load_eval_rows(episodes, selected_ids, blob):
    feature_columns = list(blob["feature_columns"])
    target_columns = list(blob["target_columns"])
    status_names = list(blob["status_names"])
    allowed_statuses = set(status_names)
    max_target_abs = dict(blob.get("max_target_abs", {}))
    rows = []
    skipped_rows = 0
    target_filter_counts = {}

    for episode in episodes:
        if selected_ids and episode["id"] not in selected_ids:
            continue

        with episode["csv_path"].open(newline="") as stream:
            reader = csv.DictReader(stream)

            for row in reader:
                if row.get("cup_status", "") not in allowed_statuses:
                    skipped_rows += 1
                    continue

                features, targets = encode_row(
                    row,
                    feature_columns,
                    target_columns,
                    status_names,
                )

                if features is None:
                    skipped_rows += 1
                    continue

                filtered_column = target_exceeds_abs(
                    targets,
                    target_columns,
                    max_target_abs,
                )

                if filtered_column is not None:
                    skipped_rows += 1
                    target_filter_counts[filtered_column] = (
                        target_filter_counts.get(filtered_column, 0) + 1
                    )
                    continue

                rows.append({
                    "episode_id": episode["id"],
                    "row_index": row.get("row_index", ""),
                    "elapsed_sec": row.get("elapsed_sec", ""),
                    "cup_status": row.get("cup_status", ""),
                    "error_x": row.get("error_x", ""),
                    "bbox_error": row.get("bbox_error", ""),
                    "theta_deg": row.get("theta_deg", ""),
                    "x": features,
                    "y": targets,
                })

    return rows, skipped_rows, target_filter_counts


def predict_rows(model, blob, rows, device):
    x = torch.tensor([row["x"] for row in rows], dtype=torch.float32)
    y = torch.tensor([row["y"] for row in rows], dtype=torch.float32)
    x_mean = torch.tensor(blob["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(blob["x_std"], dtype=torch.float32)
    y_mean = torch.tensor(blob["y_mean"], dtype=torch.float32)
    y_std = torch.tensor(blob["y_std"], dtype=torch.float32)
    x_norm = (x - x_mean) / x_std

    with torch.no_grad():
        pred_norm = model(x_norm.to(device)).cpu()

    pred = pred_norm * y_std + y_mean
    return y, pred


def target_metrics(actual, pred, target_columns, sign_eps):
    diff = pred - actual
    metrics = {}

    for index, name in enumerate(target_columns):
        target = actual[:, index]
        output = pred[:, index]
        error = diff[:, index]
        abs_error = error.abs()
        nonzero_mask = target.abs() > sign_eps

        if int(nonzero_mask.sum().item()) > 0:
            signs_match = (
                torch.sign(target[nonzero_mask])
                == torch.sign(output[nonzero_mask])
            )
            sign_agreement = float(signs_match.float().mean().item())
        else:
            sign_agreement = math.nan

        metrics[name] = {
            "mae": float(abs_error.mean().item()),
            "rmse": float(torch.sqrt((error ** 2).mean()).item()),
            "max_abs_error": float(abs_error.max().item()),
            "target_abs_mean": float(target.abs().mean().item()),
            "pred_abs_mean": float(output.abs().mean().item()),
            "nonzero_target_fraction": float(nonzero_mask.float().mean().item()),
            "sign_agreement": sign_agreement,
        }

    total_abs = diff.abs().sum(dim=1)
    return metrics, total_abs


def episode_metrics(rows, actual, pred, target_columns, sign_eps):
    by_episode = {}

    for episode_id in sorted({row["episode_id"] for row in rows}):
        indexes = [
            index
            for index, row in enumerate(rows)
            if row["episode_id"] == episode_id
        ]
        index_tensor = torch.tensor(indexes, dtype=torch.long)
        metrics, total_abs = target_metrics(
            actual[index_tensor],
            pred[index_tensor],
            target_columns,
            sign_eps,
        )
        by_episode[episode_id] = {
            "rows": len(indexes),
            "total_abs_action_error_mean": float(total_abs.mean().item()),
            "targets": metrics,
        }

    return by_episode


def worst_rows(rows, actual, pred, target_columns, total_abs, top_k):
    if top_k <= 0:
        return []

    count = min(top_k, len(rows))
    indexes = torch.topk(total_abs, k=count).indices.tolist()
    worst = []

    for index in indexes:
        item = {
            "episode_id": rows[index]["episode_id"],
            "row_index": rows[index]["row_index"],
            "elapsed_sec": rows[index]["elapsed_sec"],
            "cup_status": rows[index]["cup_status"],
            "error_x": rows[index]["error_x"],
            "bbox_error": rows[index]["bbox_error"],
            "theta_deg": rows[index]["theta_deg"],
            "total_abs_action_error": float(total_abs[index].item()),
        }

        for target_index, name in enumerate(target_columns):
            item[f"actual_{name}"] = float(actual[index, target_index].item())
            item[f"pred_{name}"] = float(pred[index, target_index].item())
            item[f"error_{name}"] = float(
                pred[index, target_index].item()
                - actual[index, target_index].item()
            )

        worst.append(item)

    return worst


def write_predictions(path, rows, actual, pred, target_columns):
    fieldnames = [
        "episode_id",
        "row_index",
        "elapsed_sec",
        "cup_status",
        "error_x",
        "bbox_error",
        "theta_deg",
    ]

    for name in target_columns:
        fieldnames.extend([
            f"actual_{name}",
            f"pred_{name}",
            f"error_{name}",
            f"abs_error_{name}",
        ])

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, row in enumerate(rows):
            output = {
                key: row[key]
                for key in (
                    "episode_id",
                    "row_index",
                    "elapsed_sec",
                    "cup_status",
                    "error_x",
                    "bbox_error",
                    "theta_deg",
                )
            }

            for target_index, name in enumerate(target_columns):
                actual_value = float(actual[row_index, target_index].item())
                pred_value = float(pred[row_index, target_index].item())
                error_value = pred_value - actual_value
                output[f"actual_{name}"] = f"{actual_value:.10f}"
                output[f"pred_{name}"] = f"{pred_value:.10f}"
                output[f"error_{name}"] = f"{error_value:.10f}"
                output[f"abs_error_{name}"] = f"{abs(error_value):.10f}"

            writer.writerow(output)


def maybe_write_plot(path, rows, actual, pred, target_columns):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    fig, axes = plt.subplots(
        len(target_columns),
        1,
        figsize=(12, max(3, len(target_columns) * 2.4)),
        sharex=True,
    )

    if len(target_columns) == 1:
        axes = [axes]

    for index, name in enumerate(target_columns):
        axes[index].plot(
            x,
            actual[:, index].tolist(),
            label="actual",
            linewidth=1.2,
        )
        axes[index].plot(
            x,
            pred[:, index].tolist(),
            label="pred",
            linewidth=1.0,
            alpha=0.85,
        )
        axes[index].set_ylabel(name)
        axes[index].grid(True, alpha=0.25)
        axes[index].legend(loc="upper right")

    axes[-1].set_xlabel("evaluation row")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    model_path = Path(args.model)
    metadata_path = (
        Path(args.metadata)
        if args.metadata is not None
        else model_path.with_name("metadata.json")
    )
    output_dir = resolve_output_dir(args)
    device = resolve_device(args.device)
    metadata = load_json(metadata_path)
    model, blob = load_model(model_path, device)
    selected_ids = selected_episode_ids(args, metadata)
    episodes = discover_episodes(Path(args.episode_root), set(args.include_label))
    rows, skipped_rows, target_filter_counts = load_eval_rows(
        episodes,
        selected_ids,
        blob,
    )

    if not rows:
        raise RuntimeError("no matching rows to evaluate")

    actual, pred = predict_rows(model, blob, rows, device)
    target_columns = list(blob["target_columns"])
    metrics, total_abs = target_metrics(
        actual,
        pred,
        target_columns,
        args.sign_eps,
    )
    per_episode = episode_metrics(
        rows,
        actual,
        pred,
        target_columns,
        args.sign_eps,
    )
    worst = worst_rows(
        rows,
        actual,
        pred,
        target_columns,
        total_abs,
        args.top_k,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    plot_path = output_dir / "timeseries.png"
    write_predictions(predictions_path, rows, actual, pred, target_columns)

    result = {
        "model": str(model_path),
        "metadata": str(metadata_path),
        "episode_root": str(args.episode_root),
        "split": args.split,
        "selected_episodes": sorted(selected_ids),
        "rows": len(rows),
        "skipped_rows": skipped_rows,
        "status_names": list(blob["status_names"]),
        "feature_columns": list(blob["feature_columns"]),
        "target_columns": target_columns,
        "max_target_abs": dict(blob.get("max_target_abs", {})),
        "target_filter_counts": target_filter_counts,
        "targets": metrics,
        "episodes": per_episode,
        "worst_rows": worst,
        "predictions_csv": str(predictions_path),
    }

    if args.plot:
        maybe_write_plot(plot_path, rows, actual, pred, target_columns)
        result["timeseries_png"] = str(plot_path)

    metrics_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"rows={len(rows)} skipped_rows={skipped_rows}")
    print(f"target_filter_counts={target_filter_counts}")
    print(f"episodes={', '.join(sorted({row['episode_id'] for row in rows}))}")
    print("metrics:")

    for name, values in metrics.items():
        print(
            f"  {name}: "
            f"mae={values['mae']:.8f} "
            f"rmse={values['rmse']:.8f} "
            f"max={values['max_abs_error']:.8f} "
            f"sign={values['sign_agreement']:.3f}"
        )

    if worst:
        print("worst_rows:")

        for item in worst[: min(5, len(worst))]:
            print(
                f"  {item['episode_id']} row={item['row_index']} "
                f"elapsed={item['elapsed_sec']} "
                f"total_abs_error={item['total_abs_action_error']:.8f}"
            )

    print("saved:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")

    if args.plot:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
