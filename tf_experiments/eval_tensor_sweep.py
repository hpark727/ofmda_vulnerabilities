#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dl_pipeline"))

import triplet_fingerprinting as tf  # noqa: E402


def metadata_for_tensor(tensor_path: Path) -> Optional[Path]:
    candidate = tensor_path.with_name(f"{tensor_path.stem}_meta.json")
    return candidate if candidate.exists() else None


def normalize_features(X: torch.Tensor, mask: torch.Tensor, normalization: dict) -> torch.Tensor:
    mean = torch.tensor(normalization["mean"], dtype=X.dtype)
    std = torch.tensor(normalization["std"], dtype=X.dtype)
    return ((X - mean) / std) * mask.unsqueeze(-1)


def evaluate_tensor(
    tensor_path: Path,
    checkpoint: dict,
    shots: List[int],
    episodes: int,
    knn_k: int,
    eval_batch_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    dataset = tf.load_dataset(tensor_path, metadata_for_tensor(tensor_path))
    dataset, empty_captures_dropped = tf.validate_and_filter_dataset(dataset)

    if dataset.X.shape[-1] != checkpoint["model_config"]["input_dim"]:
        raise ValueError(
            f"{tensor_path} has input_dim={dataset.X.shape[-1]}, but checkpoint expects "
            f"{checkpoint['model_config']['input_dim']}."
        )

    X = normalize_features(dataset.X, dataset.mask, checkpoint["normalization"])
    model = tf.TripletFingerprintNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics: Dict[str, Optional[float]] = {}
    for n_shot in shots:
        metrics[f"{n_shot}_shot_accuracy"] = tf.few_shot_accuracy(
            model=model,
            X=X,
            mask=dataset.mask,
            labels=dataset.y,
            n_shot=n_shot,
            episodes=episodes,
            knn_k=knn_k,
            batch_size=eval_batch_size,
            device=device,
            seed=seed + 1000 + n_shot,
        )

    class_counts = {
        str(label): len(indices)
        for label, indices in sorted(tf.build_class_indices(dataset.y).items())
    }
    label_names = tf.invert_label_map(dataset.label_map)
    named_class_counts = {
        label_names.get(int(label), f"class_{label}"): count
        for label, count in class_counts.items()
    }

    return {
        "tensor": str(tensor_path),
        "metadata": str(metadata_for_tensor(tensor_path))
        if metadata_for_tensor(tensor_path)
        else None,
        "empty_captures_dropped": empty_captures_dropped,
        "num_examples": int(len(dataset.y)),
        "X_shape": list(dataset.X.shape),
        "feature_names": dataset.feature_names,
        "class_counts_by_id": class_counts,
        "class_counts": named_class_counts,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved triplet weights on one or more tensor datasets."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("triplet_runs/random_negatives_only/triplet_model.pt"),
    )
    parser.add_argument(
        "--run_results",
        type=Path,
        default=Path("triplet_runs/random_negatives_only/results.json"),
        help="Saved training results JSON used for default seed, episodes, and batch size.",
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        nargs="+",
        default=None,
        help="Tensor .pt files to evaluate. Defaults to tensors/*_triplet.pt except training_captures.",
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10, 15, 20])
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("triplet_runs/random_negatives_only/tensor_sweep_k5_1_5_10_15_20.json"),
    )
    args = parser.parse_args()

    saved_results = json.loads(args.run_results.read_text())
    saved_args = saved_results["args"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    tensor_paths = args.tensors
    if tensor_paths is None:
        tensor_paths = sorted(
            path
            for path in Path("tensors").glob("*_triplet.pt")
            if path.name != "training_captures_triplet.pt"
        )
    if not tensor_paths:
        raise FileNotFoundError("No tensor datasets matched.")

    episodes = args.episodes if args.episodes is not None else saved_args["test_episodes"]
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else saved_args["eval_batch_size"]
    )
    seed = args.seed if args.seed is not None else saved_args["seed"]
    device = torch.device(args.device or saved_args.get("device", "cpu"))

    summary = {
        "checkpoint": str(args.checkpoint),
        "shots": args.shots,
        "knn_k": args.knn_k,
        "episodes": episodes,
        "eval_batch_size": eval_batch_size,
        "seed": seed,
        "device": str(device),
        "results": [],
    }

    for tensor_path in tensor_paths:
        print(f"[INFO] Evaluating {tensor_path}", file=sys.stderr)
        summary["results"].append(
            evaluate_tensor(
                tensor_path=tensor_path,
                checkpoint=checkpoint,
                shots=args.shots,
                episodes=episodes,
                knn_k=args.knn_k,
                eval_batch_size=eval_batch_size,
                device=device,
                seed=seed,
            )
        )

    output = json.dumps(summary, indent=2)
    print(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
