#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from preprocessing import build_dataset, summarize_packet_counts


DEFAULT_TARGET_MAC = "3c:64:cf:94:84:0e"
FEATURE_NAMES = [
    "relative_time",
    "delta_time",
    "packet_length",
    "direction",
]


def iter_capture_roots(data_root: Path) -> Iterable[Path]:
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            yield child


def save_tensor_dataset(
    output_prefix: Path,
    X: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    label_map: dict,
    metadata: List[dict],
) -> None:
    torch.save(
        {
            "X": X,
            "mask": mask,
            "y": y,
            "label_map": label_map,
            "feature_names": FEATURE_NAMES,
        },
        output_prefix.with_suffix(".pt"),
    )
    output_prefix.with_name(f"{output_prefix.name}_meta.json").write_text(
        json.dumps(metadata, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert each top-level capture directory under data/ into a "
            "preprocessing.py-compatible triplet tensor dataset."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Directory containing capture-set subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tensors"),
        help="Directory where .pt and metadata JSON files will be written.",
    )
    parser.add_argument(
        "--target_mac",
        type=str,
        default=DEFAULT_TARGET_MAC,
        help="MAC address to filter on. Defaults to the MAC used for earlier tensors.",
    )
    parser.add_argument(
        "--max_packets",
        type=int,
        default=3000,
        help="Pad/truncate each capture to this many packets.",
    )
    parser.add_argument(
        "--include_non_data_frames",
        action="store_true",
        help="Disable the earlier data-frame-only filter.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_triplet",
        help="Suffix appended to each capture-set name for output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tensor outputs.",
    )
    parser.add_argument(
        "--report_lengths",
        action="store_true",
        help="Also write packet-count summaries for each capture set.",
    )
    parser.add_argument(
        "--suggest_percentile",
        type=float,
        default=95.0,
        help="Percentile used for suggested max_packets in length summaries.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable per-pcap conversion progress bars.",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")
    if not args.data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {args.data_root}")
    if not 0.0 <= args.suggest_percentile <= 100.0:
        raise ValueError("--suggest_percentile must be between 0 and 100")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_roots = list(iter_capture_roots(args.data_root))
    if not capture_roots:
        raise FileNotFoundError(f"No capture-set directories found under {args.data_root}")

    manifest = {
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "target_mac": args.target_mac.lower(),
        "max_packets": args.max_packets,
        "data_only": not args.include_non_data_frames,
        "feature_names": FEATURE_NAMES,
        "datasets": [],
    }

    for capture_root in capture_roots:
        output_prefix = args.output_dir / f"{capture_root.name}{args.suffix}"
        tensor_path = output_prefix.with_suffix(".pt")
        metadata_path = output_prefix.with_name(f"{output_prefix.name}_meta.json")
        lengths_path = output_prefix.with_name(f"{output_prefix.name}_lengths.json")

        if not args.overwrite and (tensor_path.exists() or metadata_path.exists()):
            print(
                f"[SKIP] {capture_root}: output exists; pass --overwrite to regenerate.",
                file=sys.stderr,
            )
            manifest["datasets"].append(
                {
                    "capture_root": str(capture_root),
                    "tensor_path": str(tensor_path),
                    "metadata_path": str(metadata_path),
                    "status": "skipped_existing",
                }
            )
            continue

        print(f"[INFO] Converting {capture_root} -> {tensor_path}", file=sys.stderr)
        X, mask, y, label_map, metadata = build_dataset(
            input_root=capture_root,
            target_mac=args.target_mac.lower(),
            max_packets=args.max_packets,
            data_only=not args.include_non_data_frames,
            show_progress=not args.no_progress,
        )
        save_tensor_dataset(
            output_prefix=output_prefix,
            X=X,
            mask=mask,
            y=y,
            label_map=label_map,
            metadata=metadata,
        )

        length_summary = None
        if args.report_lengths:
            length_summary = summarize_packet_counts(
                metadata=metadata,
                suggest_percentile=args.suggest_percentile,
            )
            lengths_path.write_text(json.dumps(length_summary, indent=2))

        counts = np.array(
            [entry["num_packets_after_filter"] for entry in metadata],
            dtype=np.int64,
        )
        manifest_entry = {
            "capture_root": str(capture_root),
            "tensor_path": str(tensor_path),
            "metadata_path": str(metadata_path),
            "lengths_path": str(lengths_path) if args.report_lengths else None,
            "status": "converted",
            "num_captures": int(X.shape[0]),
            "X_shape": list(X.shape),
            "mask_shape": list(mask.shape),
            "num_classes": len(label_map),
            "label_map": label_map,
            "min_packets_after_filter": int(counts.min()) if counts.size else 0,
            "max_packets_after_filter": int(counts.max()) if counts.size else 0,
            "mean_packets_after_filter": float(counts.mean()) if counts.size else 0.0,
            "suggested_max_packets": (
                length_summary["suggested_max_packets"]
                if length_summary is not None
                else None
            ),
        }
        manifest["datasets"].append(manifest_entry)

        print(f"[DONE] Saved {tensor_path}", file=sys.stderr)
        print(f"[DONE] Saved {metadata_path}", file=sys.stderr)
        if args.report_lengths:
            print(f"[DONE] Saved {lengths_path}", file=sys.stderr)

    manifest_path = args.output_dir / "data_subdirs_preprocess_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
