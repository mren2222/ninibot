#!/usr/bin/env python3
"""Train a small structured imitation policy from episode CSV logs."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


DEFAULT_FEATURE_COLUMNS = [
    "error_x",
    "bbox_w",
    "bbox_error",
    "theta_deg",
    "servo_radius",
    "servo_z",
    "joint2_target_deg",
    "joint3_target_deg",
    "joint4_target_deg",
    "gripper_target_rad",
    "real_j1",
    "real_j2",
    "real_j3",
    "real_j4",
    "real_j5",
    "real_j6",
    "target_j1",
    "target_j2",
    "target_j3",
    "target_j4",
    "target_j5",
    "target_j6",
    "cmd_delta_j1",
    "cmd_delta_j2",
    "cmd_delta_j3",
    "cmd_delta_j4",
    "cmd_delta_j5",
    "cmd_delta_j6",
]

DEFAULT_TARGET_COLUMNS = [
    "action_delta_theta",
    "action_delta_joint2",
    "action_delta_joint3",
    "action_delta_gripper",
]

DEFAULT_CUP_STATUSES = [
    "SEARCH_PREP",
    "SEARCHING",
    "TRACKING_CENTER",
    "FINE_ALIGN_PREP",
    "FINE_ALIGN",
]


class StructuredPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a structured behavior-cloning policy from episodes."
    )
    parser.add_argument("--episode-root", default="data/episodes")
    parser.add_argument("--output-dir", default="data/models/structured_policy")
    parser.add_argument("--include-label", action="append", default=["success"])
    parser.add_argument(
        "--cup-status",
        action="append",
        default=None,
        help="Cup status to include. Repeatable. Defaults to active grasp statuses.",
    )
    parser.add_argument(
        "--feature-column",
        action="append",
        default=None,
        help="Feature column to use. Repeatable. Defaults to structured columns.",
    )
    parser.add_argument(
        "--target-column",
        action="append",
        default=None,
        help="Target/action column to predict. Repeatable.",
    )
    parser.add_argument(
        "--max-target-abs",
        action="append",
        default=None,
        metavar="COLUMN=VALUE",
        help=(
            "Skip rows where abs(target column) is greater than VALUE. "
            "Repeatable, for example: --max-target-abs action_delta_radius=0.003"
        ),
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. CPU is enough for this dataset.",
    )
    return parser.parse_args()


def parse_float(value):
    if value is None or value == "":
        return math.nan

    try:
        parsed = float(value)
    except ValueError:
        return math.nan

    if math.isfinite(parsed):
        return parsed

    return math.nan


def read_episode_metadata(path):
    with path.open() as stream:
        return json.load(stream)


def discover_episodes(root, include_labels):
    episodes = []

    for metadata_path in sorted(root.glob("episode_*/metadata.json")):
        metadata = read_episode_metadata(metadata_path)

        if metadata.get("label") not in include_labels:
            continue

        csv_path = metadata_path.parent / "data.csv"

        if not csv_path.exists():
            continue

        episodes.append({
            "id": metadata.get("episode_id", metadata_path.parent.name),
            "label": metadata.get("label", "unknown"),
            "csv_path": csv_path,
            "metadata_path": metadata_path,
            "rows": int(metadata.get("rows", 0)),
        })

    return episodes


def encode_row(row, feature_columns, target_columns, status_names):
    features = []

    for status in status_names:
        features.append(1.0 if row.get("cup_status", "") == status else 0.0)

    for column in feature_columns:
        value = parse_float(row.get(column))
        valid = 0.0 if math.isnan(value) else 1.0
        features.append(0.0 if math.isnan(value) else value)
        features.append(valid)

    targets = []

    for column in target_columns:
        value = parse_float(row.get(column))

        if math.isnan(value):
            return None, None

        targets.append(value)

    return features, targets


def parse_max_target_abs(items, target_columns):
    thresholds = {}
    allowed_columns = set(target_columns)

    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"invalid --max-target-abs '{item}', expected COLUMN=VALUE"
            )

        column, value_text = item.split("=", 1)
        column = column.strip()

        if column not in allowed_columns:
            raise ValueError(
                f"--max-target-abs column '{column}' is not in target columns: "
                f"{', '.join(target_columns)}"
            )

        value = parse_float(value_text.strip())

        if math.isnan(value) or value < 0.0:
            raise ValueError(
                f"--max-target-abs value for '{column}' must be a non-negative number"
            )

        thresholds[column] = value

    return thresholds


def target_exceeds_abs(targets, target_columns, max_target_abs):
    for index, column in enumerate(target_columns):
        limit = max_target_abs.get(column)

        if limit is not None and abs(targets[index]) > limit:
            return column

    return None


def load_samples(
    episodes,
    feature_columns,
    target_columns,
    status_names,
    max_target_abs=None,
):
    samples = []
    skipped_rows = 0
    status_counts = {}
    episode_counts = {}
    target_filter_counts = {}

    allowed_statuses = set(status_names)
    max_target_abs = max_target_abs or {}

    for episode in episodes:
        count = 0

        with episode["csv_path"].open(newline="") as stream:
            reader = csv.DictReader(stream)

            for row in reader:
                status = row.get("cup_status", "")

                if status not in allowed_statuses:
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

                samples.append({
                    "episode_id": episode["id"],
                    "cup_status": status,
                    "x": features,
                    "y": targets,
                })
                count += 1
                status_counts[status] = status_counts.get(status, 0) + 1

        episode_counts[episode["id"]] = count

    return (
        samples,
        skipped_rows,
        status_counts,
        episode_counts,
        target_filter_counts,
    )


def split_by_episode(samples, val_fraction, seed):
    episode_ids = sorted({sample["episode_id"] for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    val_count = max(1, int(round(len(episode_ids) * val_fraction)))
    val_ids = set(episode_ids[:val_count])

    train = [sample for sample in samples if sample["episode_id"] not in val_ids]
    val = [sample for sample in samples if sample["episode_id"] in val_ids]

    if not train or not val:
        raise RuntimeError("not enough episodes for a train/validation split")

    return train, val, sorted(val_ids)


def tensor_from_samples(samples):
    x = torch.tensor([sample["x"] for sample in samples], dtype=torch.float32)
    y = torch.tensor([sample["y"] for sample in samples], dtype=torch.float32)
    return x, y


def normalization_stats(tensor):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, x, y, y_mean, y_std, target_columns, device):
    model.eval()

    with torch.no_grad():
        pred_norm = model(x.to(device)).cpu()

    pred = pred_norm * y_std + y_mean
    diff = pred - y
    mse = float((diff ** 2).mean().item())
    mae = diff.abs().mean(dim=0)

    return {
        "mse": mse,
        "mae": {
            name: float(value.item())
            for name, value in zip(target_columns, mae)
        },
    }


def train_model(args, train_x, train_y, val_x, val_y, target_columns, device):
    x_mean, x_std = normalization_stats(train_x)
    y_mean, y_std = normalization_stats(train_y)

    train_xn = (train_x - x_mean) / x_std
    train_yn = (train_y - y_mean) / y_std
    val_xn = (val_x - x_mean) / x_std

    model = StructuredPolicy(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    loader = make_loader(train_xn, train_yn, args.batch_size, shuffle=True)

    best_state = None
    best_val_mse = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        train_metrics = evaluate(
            model,
            train_xn,
            train_y,
            y_mean,
            y_std,
            target_columns,
            device,
        )
        val_metrics = evaluate(
            model,
            val_xn,
            val_y,
            y_mean,
            y_std,
            target_columns,
            device,
        )
        item = {
            "epoch": epoch,
            "train_loss_norm": epoch_loss / max(1, batches),
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
        }
        history.append(item)

        if best_val_mse is None or val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d} "
                f"train_mse={train_metrics['mse']:.8f} "
                f"val_mse={val_metrics['mse']:.8f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = evaluate(
        model,
        train_xn,
        train_y,
        y_mean,
        y_std,
        target_columns,
        device,
    )
    final_val = evaluate(
        model,
        val_xn,
        val_y,
        y_mean,
        y_std,
        target_columns,
        device,
    )

    return model, {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
        "train_metrics": final_train,
        "val_metrics": final_val,
    }


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_artifacts(
    output_dir,
    model,
    stats,
    args,
    feature_columns,
    target_columns,
    status_names,
    episodes,
    val_episode_ids,
    sample_counts,
    max_target_abs,
    device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "policy.pt"
    metadata_path = output_dir / "metadata.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "StructuredPolicy",
            "input_dim": len(status_names) + len(feature_columns) * 2,
            "output_dim": len(target_columns),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "status_names": status_names,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "max_target_abs": max_target_abs,
            "x_mean": stats["x_mean"].tolist(),
            "x_std": stats["x_std"].tolist(),
            "y_mean": stats["y_mean"].tolist(),
            "y_std": stats["y_std"].tolist(),
        },
        model_path,
    )

    metadata = {
        "created_at_unix": time.time(),
        "episode_root": str(args.episode_root),
        "include_label": args.include_label,
        "cup_status": status_names,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "max_target_abs": max_target_abs,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "episodes": [episode["id"] for episode in episodes],
        "val_episodes": val_episode_ids,
        "sample_counts": sample_counts,
        "train_metrics": stats["train_metrics"],
        "val_metrics": stats["val_metrics"],
        "history": stats["history"],
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return model_path, metadata_path


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.episode_root = Path(args.episode_root)
    args.output_dir = Path(args.output_dir)

    feature_columns = args.feature_column or DEFAULT_FEATURE_COLUMNS
    target_columns = args.target_column or DEFAULT_TARGET_COLUMNS
    status_names = args.cup_status or DEFAULT_CUP_STATUSES
    max_target_abs = parse_max_target_abs(args.max_target_abs, target_columns)
    device = resolve_device(args.device)

    episodes = discover_episodes(args.episode_root, set(args.include_label))

    if len(episodes) < 2:
        raise RuntimeError(
            f"need at least 2 matching episodes under {args.episode_root}"
        )

    (
        samples,
        skipped_rows,
        status_counts,
        episode_counts,
        target_filter_counts,
    ) = load_samples(
        episodes,
        feature_columns,
        target_columns,
        status_names,
        max_target_abs,
    )

    if len(samples) < 32:
        raise RuntimeError(f"not enough usable samples: {len(samples)}")

    train_samples, val_samples, val_episode_ids = split_by_episode(
        samples,
        args.val_fraction,
        args.seed,
    )
    train_x, train_y = tensor_from_samples(train_samples)
    val_x, val_y = tensor_from_samples(val_samples)

    print(f"episodes={len(episodes)} samples={len(samples)} skipped_rows={skipped_rows}")
    print(f"train_samples={len(train_samples)} val_samples={len(val_samples)}")
    print(f"val_episodes={', '.join(val_episode_ids)}")
    print(f"status_counts={status_counts}")
    print(f"target_filter_counts={target_filter_counts}")
    print(f"feature_dim={train_x.shape[1]} target_dim={train_y.shape[1]}")
    print(f"device={device}")

    model, stats = train_model(
        args,
        train_x,
        train_y,
        val_x,
        val_y,
        target_columns,
        device,
    )
    model_path, metadata_path = save_artifacts(
        args.output_dir,
        model.cpu(),
        stats,
        args,
        feature_columns,
        target_columns,
        status_names,
        episodes,
        val_episode_ids,
        {
            "total": len(samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "by_status": status_counts,
            "by_episode": episode_counts,
            "skipped_rows": skipped_rows,
            "target_filter_counts": target_filter_counts,
        },
        max_target_abs,
        device,
    )

    print("saved:")
    print(f"  {model_path}")
    print(f"  {metadata_path}")
    print("validation_mae:")

    for name, value in stats["val_metrics"]["mae"].items():
        print(f"  {name}: {value:.8f}")


if __name__ == "__main__":
    main()
