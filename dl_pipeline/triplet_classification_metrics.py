#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import triplet_fingerprinting as tf


def metadata_for_tensor(tensor_path: Path) -> Optional[Path]:
    candidate = tensor_path.with_name(f"{tensor_path.stem}_meta.json")
    return candidate if candidate.exists() else None


def normalize_features(X: torch.Tensor, mask: torch.Tensor, normalization: dict) -> torch.Tensor:
    mean = torch.tensor(normalization["mean"], dtype=X.dtype)
    std = torch.tensor(normalization["std"], dtype=X.dtype)
    return ((X - mean) / std) * mask.unsqueeze(-1)


def find_feature_index(
    feature_names: Sequence[str],
    feature_dim: int,
    feature_name: str,
    fallback_index: Optional[int] = None,
) -> int:
    normalized = [name.lower() for name in feature_names]
    if feature_name.lower() in normalized:
        return normalized.index(feature_name.lower())
    if fallback_index is not None and 0 <= fallback_index < feature_dim:
        return fallback_index
    raise ValueError(
        f"Could not find feature '{feature_name}' in feature_names={list(feature_names)}."
    )


def apply_feature_ablation(
    X: torch.Tensor,
    feature_names: Sequence[str],
    ablated_features: Sequence[str],
) -> Tuple[torch.Tensor, List[dict]]:
    if not ablated_features:
        return X, []

    X = X.clone()
    applied = []
    for feature_name in ablated_features:
        fallback = 3 if feature_name == "direction" else None
        feature_idx = find_feature_index(
            feature_names=feature_names,
            feature_dim=X.shape[-1],
            feature_name=feature_name,
            fallback_index=fallback,
        )
        X[..., feature_idx] = 0.0
        applied.append({"feature": feature_name, "feature_index": feature_idx})
    return X, applied


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[WARN] Requested device '{device_name}' but CUDA is unavailable; using CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    return torch.device(device_name)


def label_names_for_ids(label_map: Dict[str, int], label_ids: Sequence[int]) -> Dict[int, str]:
    inverted = tf.invert_label_map(label_map)
    return {
        int(label_id): inverted.get(int(label_id), f"class_{int(label_id)}")
        for label_id in label_ids
    }


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
) -> List[List[int]]:
    label_to_pos = {int(label): pos for pos, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred):
        if int(true_label) not in label_to_pos or int(pred_label) not in label_to_pos:
            continue
        matrix[label_to_pos[int(true_label)]][label_to_pos[int(pred_label)]] += 1
    return matrix


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def zero_if_none(value: Optional[float]) -> float:
    return 0.0 if value is None else value


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Dict[int, str],
) -> dict:
    labels = sorted(set(int(label) for label in y_true) | set(int(label) for label in y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(labels)))

    per_class = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        fp = sum(row[i] for row in matrix) - tp
        tn = total - tp - fn - fp

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        support = tp + fn
        predicted = tp + fp
        per_class.append(
            {
                "label_id": int(label),
                "label": label_names.get(int(label), f"class_{int(label)}"),
                "support": int(support),
                "predicted": int(predicted),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "false_positive_rate": (
                    None if specificity is None else 1.0 - specificity
                ),
                "false_negative_rate": None if recall is None else 1.0 - recall,
            }
        )

    supports = [entry["support"] for entry in per_class]
    total_support = sum(supports)

    def macro_average(metric: str) -> Optional[float]:
        values = [
            entry[metric]
            for entry in per_class
            if entry[metric] is not None and entry["support"] > 0
        ]
        return None if not values else sum(values) / len(values)

    def weighted_average(metric: str) -> Optional[float]:
        if total_support == 0:
            return None
        return (
            sum(zero_if_none(entry[metric]) * entry["support"] for entry in per_class)
            / total_support
        )

    top_confusions = []
    for i, true_label in enumerate(labels):
        row_total = sum(matrix[i])
        for j, pred_label in enumerate(labels):
            if i == j or matrix[i][j] == 0:
                continue
            top_confusions.append(
                {
                    "true_label_id": int(true_label),
                    "true_label": label_names.get(int(true_label), f"class_{int(true_label)}"),
                    "predicted_label_id": int(pred_label),
                    "predicted_label": label_names.get(
                        int(pred_label),
                        f"class_{int(pred_label)}",
                    ),
                    "count": int(matrix[i][j]),
                    "rate_of_true_class": safe_divide(matrix[i][j], row_total),
                }
            )
    top_confusions.sort(key=lambda item: item["count"], reverse=True)

    return {
        "num_examples": int(total),
        "num_classes": len(labels),
        "accuracy": safe_divide(correct, total),
        "balanced_accuracy": macro_average("recall"),
        "macro_precision": macro_average("precision"),
        "macro_recall": macro_average("recall"),
        "macro_f1": macro_average("f1"),
        "weighted_precision": weighted_average("precision"),
        "weighted_recall": weighted_average("recall"),
        "weighted_f1": weighted_average("f1"),
        "micro_precision": safe_divide(correct, total),
        "micro_recall": safe_divide(correct, total),
        "micro_f1": safe_divide(correct, total),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [
                {
                    "label_id": int(label),
                    "label": label_names.get(int(label), f"class_{int(label)}"),
                }
                for label in labels
            ],
            "rows_true_cols_pred": matrix,
        },
        "top_confusions": top_confusions,
    }


def sample_support_indices(
    labels: torch.Tensor,
    n_shot: Optional[int],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    support_indices: List[int] = []
    support_labels: List[int] = []
    for label, indices in sorted(tf.build_class_indices(labels).items()):
        chosen = list(indices)
        rng.shuffle(chosen)
        support = chosen if n_shot is None else chosen[:n_shot]
        support_indices.extend(support)
        support_labels.extend([int(label)] * len(support))
    return support_indices, support_labels


def predict_with_support(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    knn_k: int,
    n_shot: Optional[int],
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    support_indices, sampled_support_labels = sample_support_indices(
        labels=support_labels,
        n_shot=n_shot,
        rng=rng,
    )
    if not support_indices:
        raise RuntimeError("No support examples are available for k-NN classification.")

    predictions = tf.knn_predict(
        query_embeddings=query_embeddings,
        support_embeddings=support_embeddings[support_indices],
        support_labels=sampled_support_labels,
        k=knn_k,
    )
    return predictions, support_indices


def predict_few_shot_episodes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_shot: int,
    episodes: int,
    knn_k: int,
    seed: int,
) -> Tuple[List[int], List[int], List[dict]]:
    class_to_indices = tf.build_class_indices(labels)
    eligible = {
        label: indices
        for label, indices in class_to_indices.items()
        if len(indices) >= n_shot + 1
    }
    if len(eligible) < 2:
        raise RuntimeError(
            f"Need at least two classes with n_shot + 1 examples; got {len(eligible)}."
        )

    rng = random.Random(seed)
    all_true: List[int] = []
    all_pred: List[int] = []
    episode_summaries: List[dict] = []

    for episode in range(1, episodes + 1):
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
            support_labels.extend([int(label)] * len(support))
            query_indices.extend(query)
            query_labels.extend([int(label)] * len(query))

        preds = tf.knn_predict(
            query_embeddings=embeddings[query_indices],
            support_embeddings=embeddings[support_indices],
            support_labels=support_labels,
            k=knn_k,
        )
        correct = sum(int(pred == gold) for pred, gold in zip(preds, query_labels))
        accuracy = correct / max(len(query_labels), 1)
        all_true.extend(query_labels)
        all_pred.extend(preds)
        episode_summaries.append(
            {
                "episode": episode,
                "support_size": len(support_indices),
                "query_size": len(query_indices),
                "accuracy": accuracy,
            }
        )

    return all_true, all_pred, episode_summaries


def summarize_episode_metrics(episode_summaries: Sequence[dict]) -> dict:
    if not episode_summaries:
        return {}
    accuracies = [float(entry["accuracy"]) for entry in episode_summaries]
    return {
        "episodes": len(episode_summaries),
        "mean_accuracy": mean(accuracies),
        "std_accuracy": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
    }


def load_saved_run(
    run_dir: Path,
    split_max_n_shot: Optional[int],
    device: torch.device,
) -> Tuple[dict, torch.nn.Module, tf.SequenceDataset, List[int], List[int]]:
    results_path = run_dir / "results.json"
    checkpoint_path = run_dir / "triplet_model.pt"
    results = json.loads(results_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_args = results["args"]

    rng = tf.set_seed(saved_args["seed"])
    dataset = tf.load_dataset(
        Path(saved_args["dataset"]),
        Path(saved_args["metadata"]) if saved_args.get("metadata") else None,
    )
    dataset, _ = tf.validate_and_filter_dataset(dataset)

    if split_max_n_shot is None:
        split_max_n_shot = max(
            saved_args["n_shot"] + [saved_args["validation_n_shot"]]
        )

    train_indices, _, test_indices, _ = tf.split_indices_per_class(
        labels=dataset.y,
        train_frac=saved_args["train_frac"],
        val_frac=saved_args["val_frac"],
        max_n_shot=split_max_n_shot,
        rng=rng,
    )
    if not train_indices or not test_indices:
        raise RuntimeError("Saved split reconstruction produced an empty train or test split.")

    model = tf.TripletFingerprintNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint, model, dataset, train_indices, test_indices


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[dict, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = tf.TripletFingerprintNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def load_tensor_dataset(tensor_path: Path, metadata_path: Optional[Path]) -> tf.SequenceDataset:
    dataset = tf.load_dataset(
        tensor_path,
        metadata_path if metadata_path is not None else metadata_for_tensor(tensor_path),
    )
    dataset, empty_captures_dropped = tf.validate_and_filter_dataset(dataset)
    if empty_captures_dropped:
        print(
            f"[INFO] Dropped {empty_captures_dropped} empty captures from {tensor_path}.",
            file=sys.stderr,
        )
    return dataset


def make_embeddings(
    model: torch.nn.Module,
    dataset: tf.SequenceDataset,
    indices: Sequence[int],
    normalization: dict,
    eval_batch_size: int,
    device: torch.device,
    ablated_features: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    X, mask, labels = tf.subset_tensor_dataset(
        dataset.X,
        dataset.mask,
        dataset.y,
        indices,
    )
    X = normalize_features(X, mask, normalization)
    X, _ = apply_feature_ablation(X, dataset.feature_names, ablated_features)
    embeddings = tf.compute_embeddings(
        model,
        X,
        mask,
        batch_size=eval_batch_size,
        device=device,
    )
    return embeddings, labels


def make_all_embeddings(
    model: torch.nn.Module,
    dataset: tf.SequenceDataset,
    normalization: dict,
    eval_batch_size: int,
    device: torch.device,
    ablated_features: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = list(range(dataset.X.shape[0]))
    return make_embeddings(
        model=model,
        dataset=dataset,
        indices=indices,
        normalization=normalization,
        eval_batch_size=eval_batch_size,
        device=device,
        ablated_features=ablated_features,
    )


def compact_console_summary(report: dict, top_k: int) -> str:
    def fmt(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        f"Accuracy:          {fmt(report['accuracy'])}",
        f"Balanced accuracy: {fmt(report['balanced_accuracy'])}",
        f"Macro F1:          {fmt(report['macro_f1'])}",
        f"Weighted F1:       {fmt(report['weighted_f1'])}",
    ]
    if report["top_confusions"]:
        lines.append("")
        lines.append("Top confusions:")
        for item in report["top_confusions"][:top_k]:
            lines.append(
                "  "
                f"{item['true_label']} -> {item['predicted_label']}: "
                f"{item['count']} ({item['rate_of_true_class']:.2%})"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report confusion matrix, precision, recall, F1, and related "
            "classification metrics for a saved triplet fingerprinting model."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Directory containing results.json and triplet_model.pt.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Direct path to a triplet_model.pt checkpoint. Defaults to run_dir/triplet_model.pt.",
    )
    parser.add_argument(
        "--run_results",
        type=Path,
        default=None,
        help=(
            "Direct path to a results.json file for defaults such as seed, k, and batch size. "
            "Defaults to run_dir/results.json when run_dir is supplied."
        ),
    )
    parser.add_argument(
        "--query_tensor",
        type=Path,
        default=None,
        help="Labeled tensor dataset to classify/evaluate.",
    )
    parser.add_argument(
        "--query_metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON for query_tensor. Defaults to *_meta.json beside the tensor.",
    )
    parser.add_argument(
        "--support_tensor",
        type=Path,
        default=None,
        help=(
            "Optional labeled tensor dataset to use as k-NN support. "
            "If omitted with --query_tensor, use --mode test_few_shot."
        ),
    )
    parser.add_argument(
        "--support_metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON for support_tensor. Defaults to *_meta.json beside the tensor.",
    )
    parser.add_argument(
        "--mode",
        choices=["train_support_test_query", "test_few_shot"],
        default="train_support_test_query",
        help=(
            "Use the saved train split as k-NN support and saved test split as query, "
            "or reproduce few-shot episodes inside the test split."
        ),
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=None,
        help=(
            "Optional number of support examples per class. In train_support_test_query "
            "mode, omit this to use all train support examples. In test_few_shot mode, "
            "omit this to use 1-shot."
        ),
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--knn_k", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--ignore_direction_feature",
        action="store_true",
        help=(
            "Ablate the direction channel at evaluation time by setting the normalized "
            "direction feature to 0 before computing embeddings."
        ),
    )
    parser.add_argument(
        "--ablate_features",
        nargs="+",
        default=[],
        help=(
            "Feature names to ablate at evaluation time by setting their normalized "
            "channels to 0 before computing embeddings. Example: --ablate_features "
            "relative_time delta_time packet_length direction"
        ),
    )
    parser.add_argument(
        "--split_max_n_shot",
        type=int,
        default=None,
        help="Max n-shot value used when reconstructing the saved split.",
    )
    parser.add_argument(
        "--top_confusions",
        type=int,
        default=10,
        help="Number of largest off-diagonal confusion pairs to print.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_results_path = args.run_results
    if run_results_path is None and args.run_dir is not None:
        run_results_path = args.run_dir / "results.json"
    saved_args = {}
    if run_results_path is not None and run_results_path.exists():
        saved_results = json.loads(run_results_path.read_text())
        saved_args = saved_results.get("args", {})

    device = resolve_device(args.device or saved_args.get("device", "cpu"))
    eval_batch_size = args.eval_batch_size or saved_args.get("eval_batch_size", 256)
    knn_k = args.knn_k if args.knn_k is not None else saved_args.get("knn_k", 1)
    seed = args.seed if args.seed is not None else saved_args.get("seed", 42)
    ablated_features = list(args.ablate_features)
    if args.ignore_direction_feature and "direction" not in ablated_features:
        ablated_features.append("direction")

    episode_summaries: List[dict] = []
    source_summary = {}

    if args.query_tensor is not None:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            if args.run_dir is None:
                raise ValueError("Use --checkpoint or --run_dir when evaluating --query_tensor.")
            checkpoint_path = args.run_dir / "triplet_model.pt"
        checkpoint, model = load_checkpoint(checkpoint_path, device)

        query_dataset = load_tensor_dataset(args.query_tensor, args.query_metadata)
        if query_dataset.X.shape[-1] != checkpoint["model_config"]["input_dim"]:
            raise ValueError(
                f"{args.query_tensor} has input_dim={query_dataset.X.shape[-1]}, but "
                f"checkpoint expects {checkpoint['model_config']['input_dim']}."
            )
        test_embeddings, test_labels = make_all_embeddings(
            model=model,
            dataset=query_dataset,
            normalization=checkpoint["normalization"],
            eval_batch_size=eval_batch_size,
            device=device,
            ablated_features=ablated_features,
        )
        label_ids = sorted(
            set(int(label) for label in query_dataset.y.tolist())
            | set(int(label) for label in query_dataset.label_map.values())
        )
        label_names = label_names_for_ids(query_dataset.label_map, label_ids)
        source_summary = {
            "checkpoint": str(checkpoint_path),
            "query_tensor": str(args.query_tensor),
            "query_metadata": str(args.query_metadata or metadata_for_tensor(args.query_tensor)),
            "query_size": int(len(query_dataset.y)),
            "query_shape": list(query_dataset.X.shape),
        }

        if args.mode == "train_support_test_query":
            if args.support_tensor is None:
                raise ValueError(
                    "--mode train_support_test_query with --query_tensor requires --support_tensor. "
                    "Use --mode test_few_shot to evaluate episodes inside the query tensor."
                )
            support_dataset = load_tensor_dataset(args.support_tensor, args.support_metadata)
            if support_dataset.X.shape[-1] != checkpoint["model_config"]["input_dim"]:
                raise ValueError(
                    f"{args.support_tensor} has input_dim={support_dataset.X.shape[-1]}, but "
                    f"checkpoint expects {checkpoint['model_config']['input_dim']}."
                )
            train_embeddings, train_labels = make_all_embeddings(
                model=model,
                dataset=support_dataset,
                normalization=checkpoint["normalization"],
                eval_batch_size=eval_batch_size,
                device=device,
                ablated_features=ablated_features,
            )
            source_summary.update(
                {
                    "support_tensor": str(args.support_tensor),
                    "support_metadata": str(
                        args.support_metadata or metadata_for_tensor(args.support_tensor)
                    ),
                    "support_size": int(len(support_dataset.y)),
                    "support_shape": list(support_dataset.X.shape),
                }
            )
        else:
            train_embeddings = torch.empty(0)
            train_labels = torch.empty(0, dtype=torch.long)
    else:
        if args.run_dir is None:
            raise ValueError("Use --run_dir, or use --checkpoint with --query_tensor.")
        checkpoint, model, dataset, train_indices, test_indices = load_saved_run(
            run_dir=args.run_dir,
            split_max_n_shot=args.split_max_n_shot,
            device=device,
        )
        label_ids = sorted(int(label) for label in dataset.label_map.values())
        label_names = label_names_for_ids(dataset.label_map, label_ids)

        train_embeddings, train_labels = make_embeddings(
            model=model,
            dataset=dataset,
            indices=train_indices,
            normalization=checkpoint["normalization"],
            eval_batch_size=eval_batch_size,
            device=device,
            ablated_features=ablated_features,
        )
        test_embeddings, test_labels = make_embeddings(
            model=model,
            dataset=dataset,
            indices=test_indices,
            normalization=checkpoint["normalization"],
            eval_batch_size=eval_batch_size,
            device=device,
            ablated_features=ablated_features,
        )
        source_summary = {
            "run_dir": str(args.run_dir),
            "checkpoint": str(args.run_dir / "triplet_model.pt"),
            "train_size": len(train_indices),
            "test_size": len(test_indices),
        }

    if args.mode == "train_support_test_query":
        all_true: List[int] = []
        all_pred: List[int] = []
        episodes = max(args.episodes, 1) if args.n_shot is not None else 1
        for episode in range(1, episodes + 1):
            preds, support_indices = predict_with_support(
                support_embeddings=train_embeddings,
                support_labels=train_labels,
                query_embeddings=test_embeddings,
                knn_k=knn_k,
                n_shot=args.n_shot,
                seed=seed + episode,
            )
            query_labels = [int(label) for label in test_labels.tolist()]
            correct = sum(int(pred == gold) for pred, gold in zip(preds, query_labels))
            all_true.extend(query_labels)
            all_pred.extend(preds)
            episode_summaries.append(
                {
                    "episode": episode,
                    "support_size": len(support_indices),
                    "query_size": len(query_labels),
                    "accuracy": correct / max(len(query_labels), 1),
                }
            )
    else:
        n_shot = args.n_shot if args.n_shot is not None else 1
        all_true, all_pred, episode_summaries = predict_few_shot_episodes(
            embeddings=test_embeddings,
            labels=test_labels,
            n_shot=n_shot,
            episodes=args.episodes,
            knn_k=knn_k,
            seed=seed,
        )

    report = classification_report(all_true, all_pred, label_names)
    summary = {
        **source_summary,
        "mode": args.mode,
        "n_shot": args.n_shot,
        "episodes": len(episode_summaries),
        "knn_k": knn_k,
        "eval_batch_size": eval_batch_size,
        "seed": seed,
        "device": str(device),
        "ablated_features": ablated_features,
        "episode_metrics": summarize_episode_metrics(episode_summaries),
        "episode_summaries": episode_summaries,
        "metrics": report,
    }

    print(compact_console_summary(report, top_k=args.top_confusions))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nSaved classification metrics to: {args.output}")
    else:
        print("\nFull JSON report:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
