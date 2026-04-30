#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dl_pipeline"))

import triplet_fingerprinting as tf  # noqa: E402


def accuracy_from_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_shot: int,
    episodes: int,
    knn_k: int,
    seed: int,
) -> Optional[float]:
    class_to_indices = tf.build_class_indices(labels)
    eligible = {
        label: indices
        for label, indices in class_to_indices.items()
        if len(indices) >= n_shot + 1
    }
    if len(eligible) < 2:
        return None

    rng = random.Random(seed)
    episode_scores: List[float] = []
    for _ in range(episodes):
        support_indices: List[int] = []
        support_labels: List[int] = []
        query_indices: List[int] = []
        query_labels: List[int] = []

        for label, indices in sorted(eligible.items()):
            chosen = list(indices)
            rng.shuffle(chosen)
            support = chosen[:n_shot]
            query = chosen[n_shot:]
            support_indices.extend(support)
            support_labels.extend([label] * len(support))
            query_indices.extend(query)
            query_labels.extend([label] * len(query))

        preds = tf.knn_predict(
            query_embeddings=embeddings[query_indices],
            support_embeddings=embeddings[support_indices],
            support_labels=support_labels,
            k=knn_k,
        )
        correct = sum(int(pred == gold) for pred, gold in zip(preds, query_labels))
        episode_scores.append(correct / max(len(query_labels), 1))

    return sum(episode_scores) / len(episode_scores)


def normalize_features(X: torch.Tensor, mask: torch.Tensor, normalization: dict) -> torch.Tensor:
    mean = torch.tensor(normalization["mean"], dtype=X.dtype)
    std = torch.tensor(normalization["std"], dtype=X.dtype)
    return ((X - mean) / std) * mask.unsqueeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a saved triplet model on the saved test split."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("triplet_runs/random_negatives_only"),
        help="Directory containing results.json and triplet_model.pt.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[1, 5, 10, 15, 20],
        help="n-shot support sizes to evaluate.",
    )
    parser.add_argument(
        "--split_max_n_shot",
        type=int,
        default=None,
        help=(
            "Max n-shot value used when reconstructing the original split. "
            "Defaults to the max saved in the training args."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the evaluation summary.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=None,
        help="Override the saved k value used for nearest-neighbor voting.",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    results_path = args.run_dir / "results.json"
    checkpoint_path = args.run_dir / "triplet_model.pt"
    results = json.loads(results_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_args = results["args"]

    device = torch.device(args.device or saved_args.get("device", "cpu"))
    rng = tf.set_seed(saved_args["seed"])
    dataset = tf.load_dataset(
        Path(saved_args["dataset"]),
        Path(saved_args["metadata"]) if saved_args.get("metadata") else None,
    )
    dataset, empty_captures_dropped = tf.validate_and_filter_dataset(dataset)

    split_max_n_shot = args.split_max_n_shot
    if split_max_n_shot is None:
        split_max_n_shot = max(
            saved_args["n_shot"] + [saved_args["validation_n_shot"]]
        )

    _, _, test_indices, dropped = tf.split_indices_per_class(
        labels=dataset.y,
        train_frac=saved_args["train_frac"],
        val_frac=saved_args["val_frac"],
        max_n_shot=split_max_n_shot,
        rng=rng,
    )
    test_X, test_mask, test_y = tf.subset_tensor_dataset(
        dataset.X,
        dataset.mask,
        dataset.y,
        test_indices,
    )
    test_X = normalize_features(test_X, test_mask, checkpoint["normalization"])

    model = tf.TripletFingerprintNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    embeddings = tf.compute_embeddings(
        model,
        test_X,
        test_mask,
        batch_size=saved_args["eval_batch_size"],
        device=device,
    )

    knn_k = args.knn_k if args.knn_k is not None else saved_args["knn_k"]
    metrics: Dict[str, Optional[float]] = {}
    for n_shot in args.shots:
        metrics[f"{n_shot}_shot_accuracy"] = accuracy_from_embeddings(
            embeddings=embeddings,
            labels=test_y,
            n_shot=n_shot,
            episodes=saved_args["test_episodes"],
            knn_k=knn_k,
            seed=saved_args["seed"] + 1000 + n_shot,
        )

    class_counts = {
        str(label): len(indices)
        for label, indices in sorted(tf.build_class_indices(test_y).items())
    }
    summary = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "preserved_split_max_n_shot": split_max_n_shot,
        "empty_captures_dropped": empty_captures_dropped,
        "dropped_classes": dropped,
        "test_size": int(len(test_y)),
        "test_examples_per_class": class_counts,
        "test_episodes": saved_args["test_episodes"],
        "knn_k": knn_k,
        "metrics": metrics,
    }

    output = json.dumps(summary, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
