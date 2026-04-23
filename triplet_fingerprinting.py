#!/usr/bin/env python3

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> random.Random:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


@dataclass
class SequenceDataset:
    X: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    feature_names: List[str]
    label_map: Dict[str, int]
    metadata: Optional[List[dict]]


class MaskedFeatureStandardizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X: torch.Tensor, mask: torch.Tensor) -> "MaskedFeatureStandardizer":
        weights = mask.unsqueeze(-1)
        denom = weights.sum(dim=(0, 1)).clamp_min(1.0)
        mean = (X * weights).sum(dim=(0, 1)) / denom
        var = (((X - mean) * weights) ** 2).sum(dim=(0, 1)) / denom
        std = torch.sqrt(var.clamp_min(1e-6))
        return cls(mean=mean, std=std)

    def transform(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = (X - self.mean) / self.std
        return normalized * mask.unsqueeze(-1)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        if x.shape[-1] >= 4:
            x = self.pool(x)
        return self.dropout(x)


class TripletFingerprintNet(nn.Module):
    """
    Paper-inspired sequence encoder adapted for multivariate packet features.
    The original TF paper uses the DF backbone on +/-1 direction traces.
    Here we apply the same metric-learning idea to [T, F] packet feature vectors.
    """

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvBlock(input_dim, 64, dropout),
                ConvBlock(64, 128, dropout),
                ConvBlock(128, 256, dropout),
            ]
        )
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = X.transpose(1, 2)
        pooled_mask = mask.float()
        x = x * pooled_mask.unsqueeze(1)

        for block in self.blocks:
            before_block_len = x.shape[-1]
            x = block(x)
            if before_block_len >= 4:
                pooled_mask = F.max_pool1d(
                    pooled_mask.unsqueeze(1),
                    kernel_size=4,
                    stride=4,
                ).squeeze(1)
            x = x * pooled_mask.unsqueeze(1)

        denom = pooled_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * pooled_mask.unsqueeze(1)).sum(dim=-1) / denom
        embedding = self.proj(pooled)
        return F.normalize(embedding, p=2, dim=-1)


def load_dataset(dataset_path: Path, metadata_path: Optional[Path]) -> SequenceDataset:
    payload = torch.load(dataset_path, map_location="cpu")
    X = payload["X"].float()
    mask = payload["mask"].float()
    y = payload["y"].long()
    feature_names = list(payload.get("feature_names", []))
    label_map = dict(payload.get("label_map", {}))

    metadata = None
    if metadata_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return SequenceDataset(
        X=X,
        mask=mask,
        y=y,
        feature_names=feature_names,
        label_map=label_map,
        metadata=metadata,
    )


def build_class_indices(labels: torch.Tensor) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        class_to_indices[int(label)].append(idx)
    return dict(class_to_indices)


def allocate_split_counts(
    n: int,
    train_frac: float,
    val_frac: float,
    max_n_shot: int,
) -> Optional[Tuple[int, int, int]]:
    train_min = 2
    val_min = 1 if val_frac > 0 else 0
    test_min = max_n_shot + 1
    required = train_min + val_min + test_min
    if n < required:
        return None

    mins = {"train": train_min, "val": val_min, "test": test_min}
    weights = {
        "train": max(train_frac, 0.0),
        "val": max(val_frac, 0.0),
        "test": max(1.0 - train_frac - val_frac, 0.0),
    }
    total_weight = sum(weights.values()) or 1.0
    remaining = n - required

    extras = {
        split: int(math.floor(remaining * weights[split] / total_weight))
        for split in mins
    }
    leftover = remaining - sum(extras.values())
    order = sorted(mins, key=lambda split: weights[split], reverse=True)
    for i in range(leftover):
        extras[order[i % len(order)]] += 1

    n_train = mins["train"] + extras["train"]
    n_val = mins["val"] + extras["val"]
    n_test = mins["test"] + extras["test"]
    return n_train, n_val, n_test


def split_indices_per_class(
    labels: torch.Tensor,
    train_frac: float,
    val_frac: float,
    max_n_shot: int,
    rng: random.Random,
) -> Tuple[List[int], List[int], List[int], Dict[str, int]]:
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    dropped: Dict[str, int] = {}

    for label, indices in sorted(build_class_indices(labels).items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        counts = allocate_split_counts(
            n=len(shuffled),
            train_frac=train_frac,
            val_frac=val_frac,
            max_n_shot=max_n_shot,
        )
        if counts is None:
            dropped[str(label)] = len(indices)
            continue

        n_train, n_val, n_test = counts
        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train:n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    return train_indices, val_indices, test_indices, dropped


def subset_tensor_dataset(
    X: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return X[index_tensor], mask[index_tensor], y[index_tensor]


def build_positive_pairs(
    labels: torch.Tensor,
    rng: random.Random,
    max_pairs_per_class: Optional[int],
) -> Tuple[List[int], List[int]]:
    anchors: List[int] = []
    positives: List[int] = []

    for _, indices in sorted(build_class_indices(labels).items()):
        if len(indices) < 2:
            continue

        pairs = list(combinations(indices, 2))
        if max_pairs_per_class is not None and len(pairs) > max_pairs_per_class:
            pairs = rng.sample(pairs, max_pairs_per_class)
        else:
            rng.shuffle(pairs)

        for anchor_idx, positive_idx in pairs:
            if rng.random() < 0.5:
                anchor_idx, positive_idx = positive_idx, anchor_idx
            anchors.append(anchor_idx)
            positives.append(positive_idx)

    order = list(range(len(anchors)))
    rng.shuffle(order)
    anchors = [anchors[i] for i in order]
    positives = [positives[i] for i in order]
    return anchors, positives


@torch.no_grad()
def compute_embeddings(
    model: nn.Module,
    X: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    embeddings = []
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_X = X[start:end].to(device)
        batch_mask = mask[start:end].to(device)
        embeddings.append(model(batch_X, batch_mask).cpu())
    return torch.cat(embeddings, dim=0)


def mine_negatives(
    anchor_indices: Sequence[int],
    positive_indices: Sequence[int],
    labels: torch.Tensor,
    similarities: Optional[torch.Tensor],
    margin: float,
    rng: random.Random,
) -> List[int]:
    class_to_indices = build_class_indices(labels)
    negatives: List[int] = []

    for anchor_idx, positive_idx in zip(anchor_indices, positive_indices):
        anchor_label = int(labels[anchor_idx].item())
        candidate_pool = [
            idx for label, idxs in class_to_indices.items() if label != anchor_label for idx in idxs
        ]

        if not candidate_pool:
            raise RuntimeError("Need at least two classes to mine triplets.")

        if similarities is None:
            negatives.append(rng.choice(candidate_pool))
            continue

        pos_sim = float(similarities[anchor_idx, positive_idx].item())
        viable = [
            idx
            for idx in candidate_pool
            if float(similarities[anchor_idx, idx].item()) + margin > pos_sim
        ]

        if viable:
            negatives.append(rng.choice(viable))
            continue

        hardest = max(
            candidate_pool,
            key=lambda idx: float(similarities[anchor_idx, idx].item()),
        )
        negatives.append(hardest)

    return negatives


def cosine_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, float, float]:
    pos_sim = (anchor * positive).sum(dim=-1)
    neg_sim = (anchor * negative).sum(dim=-1)
    loss = torch.relu(neg_sim - pos_sim + margin).mean()
    return loss, float(pos_sim.mean().item()), float(neg_sim.mean().item())


def iterate_triplet_batches(
    X: torch.Tensor,
    mask: torch.Tensor,
    anchor_indices: Sequence[int],
    positive_indices: Sequence[int],
    negative_indices: Sequence[int],
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    for start in range(0, len(anchor_indices), batch_size):
        end = start + batch_size
        a_idx = torch.tensor(anchor_indices[start:end], dtype=torch.long)
        p_idx = torch.tensor(positive_indices[start:end], dtype=torch.long)
        n_idx = torch.tensor(negative_indices[start:end], dtype=torch.long)
        yield (
            X[a_idx],
            mask[a_idx],
            X[p_idx],
            mask[p_idx],
            X[n_idx],
            mask[n_idx],
        )


def knn_predict(
    query_embeddings: torch.Tensor,
    support_embeddings: torch.Tensor,
    support_labels: Sequence[int],
    k: int,
) -> List[int]:
    sims = query_embeddings @ support_embeddings.T
    k = min(k, support_embeddings.shape[0])
    neighbor_indices = sims.topk(k=k, dim=1).indices

    predictions: List[int] = []
    for query_idx, row in enumerate(neighbor_indices.tolist()):
        neighbor_labels = [support_labels[idx] for idx in row]
        counts = Counter(neighbor_labels)
        best_label = max(
            counts,
            key=lambda label: (
                counts[label],
                sum(
                    float(sims[query_idx, support_idx].item())
                    for support_idx in row
                    if support_labels[support_idx] == label
                ),
            ),
        )
        predictions.append(best_label)
    return predictions


def serialize_args(args: argparse.Namespace) -> dict:
    serialized = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def few_shot_accuracy(
    model: nn.Module,
    X: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    n_shot: int,
    episodes: int,
    knn_k: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Optional[float]:
    class_to_indices = build_class_indices(labels)
    eligible = {
        label: indices
        for label, indices in class_to_indices.items()
        if len(indices) >= n_shot + 1
    }
    if len(eligible) < 2:
        return None

    embeddings = compute_embeddings(model, X, mask, batch_size=batch_size, device=device)
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

        preds = knn_predict(
            query_embeddings=embeddings[query_indices],
            support_embeddings=embeddings[support_indices],
            support_labels=support_labels,
            k=knn_k,
        )
        correct = sum(int(pred == gold) for pred, gold in zip(preds, query_labels))
        episode_scores.append(correct / max(len(query_labels), 1))

    return sum(episode_scores) / len(episode_scores)


def train_triplet_model(
    train_X: torch.Tensor,
    train_mask: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_mask: torch.Tensor,
    val_y: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[nn.Module, List[dict]]:
    device = torch.device(args.device)
    model = TripletFingerprintNet(
        input_dim=train_X.shape[-1],
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )

    rng = random.Random(args.seed)
    history: List[dict] = []
    best_state = None
    best_val = -1.0
    similarities = None

    for epoch in range(1, args.epochs + 1):
        anchors, positives = build_positive_pairs(
            labels=train_y,
            rng=rng,
            max_pairs_per_class=args.max_pairs_per_class,
        )
        if not anchors:
            raise RuntimeError("No positive pairs could be created from the training split.")

        negatives = mine_negatives(
            anchor_indices=anchors,
            positive_indices=positives,
            labels=train_y,
            similarities=similarities,
            margin=args.margin,
            rng=rng,
        )

        model.train()
        epoch_loss = 0.0
        epoch_pos_sim = 0.0
        epoch_neg_sim = 0.0
        num_batches = 0

        for xa, ma, xp, mp, xn, mn in iterate_triplet_batches(
            X=train_X,
            mask=train_mask,
            anchor_indices=anchors,
            positive_indices=positives,
            negative_indices=negatives,
            batch_size=args.batch_size,
        ):
            xa = xa.to(device)
            ma = ma.to(device)
            xp = xp.to(device)
            mp = mp.to(device)
            xn = xn.to(device)
            mn = mn.to(device)

            optimizer.zero_grad(set_to_none=True)
            emb_a = model(xa, ma)
            emb_p = model(xp, mp)
            emb_n = model(xn, mn)
            loss, pos_sim, neg_sim = cosine_triplet_loss(
                emb_a,
                emb_p,
                emb_n,
                margin=args.margin,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_pos_sim += pos_sim
            epoch_neg_sim += neg_sim
            num_batches += 1

        similarities = compute_embeddings(
            model,
            train_X,
            train_mask,
            batch_size=args.eval_batch_size,
            device=device,
        )
        similarities = similarities @ similarities.T

        if len(val_y) > 0:
            val_acc = few_shot_accuracy(
                model=model,
                X=val_X,
                mask=val_mask,
                labels=val_y,
                n_shot=args.validation_n_shot,
                episodes=args.validation_episodes,
                knn_k=args.knn_k,
                batch_size=args.eval_batch_size,
                device=device,
                seed=args.seed + epoch,
            )
        else:
            val_acc = None

        history_entry = {
            "epoch": epoch,
            "loss": epoch_loss / max(num_batches, 1),
            "mean_positive_similarity": epoch_pos_sim / max(num_batches, 1),
            "mean_negative_similarity": epoch_neg_sim / max(num_batches, 1),
            "validation_accuracy": val_acc,
            "num_triplets": len(anchors),
        }
        history.append(history_entry)
        print(json.dumps(history_entry))

        score = -1.0 if val_acc is None else val_acc
        if score >= best_val:
            best_val = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {int(idx): name for name, idx in label_map.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Tensor dataset produced by preprocessing.py (the .pt file).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON produced by preprocessing.py.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("triplet_runs"))
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--max_pairs_per_class",
        type=int,
        default=256,
        help="Cap the number of anchor/positive pairs sampled per class each epoch.",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Few-shot support sizes to evaluate on the test split.",
    )
    parser.add_argument("--validation_n_shot", type=int, default=1)
    parser.add_argument("--validation_episodes", type=int, default=10)
    parser.add_argument("--test_episodes", type=int, default=25)
    parser.add_argument("--knn_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = set_seed(args.seed)
    dataset = load_dataset(args.dataset, args.metadata)
    max_n_shot = max(args.n_shot + [args.validation_n_shot])

    train_indices, val_indices, test_indices, dropped = split_indices_per_class(
        labels=dataset.y,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        max_n_shot=max_n_shot,
        rng=rng,
    )

    if not train_indices or not test_indices:
        raise RuntimeError(
            "Split failed: need enough traces per class for training and few-shot testing."
        )

    train_X, train_mask, train_y = subset_tensor_dataset(
        dataset.X,
        dataset.mask,
        dataset.y,
        train_indices,
    )
    val_X, val_mask, val_y = subset_tensor_dataset(
        dataset.X,
        dataset.mask,
        dataset.y,
        val_indices,
    ) if val_indices else (
        dataset.X.new_zeros((0, dataset.X.shape[1], dataset.X.shape[2])),
        dataset.mask.new_zeros((0, dataset.mask.shape[1])),
        dataset.y.new_zeros((0,), dtype=torch.long),
    )
    test_X, test_mask, test_y = subset_tensor_dataset(
        dataset.X,
        dataset.mask,
        dataset.y,
        test_indices,
    )

    standardizer = MaskedFeatureStandardizer.fit(train_X, train_mask)
    train_X = standardizer.transform(train_X, train_mask)
    val_X = standardizer.transform(val_X, val_mask)
    test_X = standardizer.transform(test_X, test_mask)

    model, history = train_triplet_model(
        train_X=train_X,
        train_mask=train_mask,
        train_y=train_y,
        val_X=val_X,
        val_mask=val_mask,
        val_y=val_y,
        args=args,
    )

    test_metrics = {}
    for n_shot in args.n_shot:
        acc = few_shot_accuracy(
            model=model,
            X=test_X,
            mask=test_mask,
            labels=test_y,
            n_shot=n_shot,
            episodes=args.test_episodes,
            knn_k=args.knn_k,
            batch_size=args.eval_batch_size,
            device=torch.device(args.device),
            seed=args.seed + 1000 + n_shot,
        )
        test_metrics[f"{n_shot}_shot_accuracy"] = acc

    label_names = invert_label_map(dataset.label_map)
    run_summary = {
        "dataset": str(args.dataset),
        "metadata": str(args.metadata) if args.metadata else None,
        "feature_names": dataset.feature_names,
        "train_size": len(train_y),
        "val_size": len(val_y),
        "test_size": len(test_y),
        "num_classes_train": len(build_class_indices(train_y)),
        "num_classes_test": len(build_class_indices(test_y)),
        "dropped_classes": {
            label_names.get(int(label), f"class_{label}"): count
            for label, count in dropped.items()
        },
        "history": history,
        "test_metrics": test_metrics,
        "normalization": standardizer.state_dict(),
        "args": serialize_args(args),
    }

    checkpoint_path = args.output_dir / "triplet_model.pt"
    results_path = args.output_dir / "results.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": train_X.shape[-1],
                "embedding_dim": args.embedding_dim,
                "dropout": args.dropout,
            },
            "normalization": standardizer.state_dict(),
            "feature_names": dataset.feature_names,
            "label_map": dataset.label_map,
            "args": serialize_args(args),
        },
        checkpoint_path,
    )
    results_path.write_text(json.dumps(run_summary, indent=2))

    print(f"Saved model checkpoint to: {checkpoint_path}")
    print(f"Saved run summary to:     {results_path}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
