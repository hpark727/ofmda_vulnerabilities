#!/usr/bin/env python3

import argparse
import json
import sys
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "wlan.sa",
    "wlan.da",
]


def print_progress(current: int, total: int, succeeded: int, failed: int, width: int = 30):
    """
    Print an in-place ASCII progress bar for pcap tensor conversion.
    """
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100.0
    message = (
        f"\rConverting pcaps [{bar}] {percent:6.2f}% "
        f"({current}/{total}, ok={succeeded}, failed={failed})"
    )
    print(message, end="", file=sys.stderr, flush=True)

    if current >= total:
        print(file=sys.stderr, flush=True)


def build_display_filter(target_mac: str, data_only: bool) -> str:
    """
    Keep packets where the target MAC is either source or destination.
    Optionally keep only data frames.
    """
    mac_filter = f'(wlan.sa == {target_mac} || wlan.da == {target_mac})'
    if data_only:
        return f'({mac_filter}) && (wlan.fc.type == 2)'
    return mac_filter


def run_tshark(pcap_path: Path, display_filter: str) -> pd.DataFrame:
    """
    Use tshark to extract selected fields into a DataFrame.
    """
    cmd = [
        "tshark",
        "-r", str(pcap_path),
        "-Y", display_filter,
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f",
    ]

    for f in FIELDS:
        cmd.extend(["-e", f])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    stdout = result.stdout.strip()
    if not stdout:
        return pd.DataFrame(columns=FIELDS)

    df = pd.read_csv(StringIO(stdout))
    return df


def df_to_packet_tensor(df: pd.DataFrame, max_packets: int, target_mac: str):
    """
    Convert one filtered capture DataFrame into:
      X:    [max_packets, 4]
      mask: [max_packets]
    features = [relative_time, delta_time, packet_length, direction]
    """
    X = np.zeros((max_packets, 4), dtype=np.float32)
    mask = np.zeros((max_packets,), dtype=np.float32)

    if df.empty:
        return torch.tensor(X), torch.tensor(mask)

    # clean numeric columns
    for col in ["frame.time_epoch", "frame.len"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frame.time_epoch", "frame.len"]).copy()
    if df.empty:
        return torch.tensor(X), torch.tensor(mask)

    df = df.sort_values("frame.time_epoch").reset_index(drop=True)

    # relative time
    t = df["frame.time_epoch"].to_numpy(dtype=np.float64)
    rel_t = t - t[0]

    # delta time
    delta_t = np.zeros_like(rel_t, dtype=np.float64)
    delta_t[1:] = rel_t[1:] - rel_t[:-1]

    # packet length
    pkt_len = df["frame.len"].to_numpy(dtype=np.float32)

    target_mac = target_mac.lower()
    source = df["wlan.sa"].fillna("").astype(str).str.lower()
    destination = df["wlan.da"].fillna("").astype(str).str.lower()
    direction = np.where(
        source == target_mac,
        1.0,
        np.where(destination == target_mac, -1.0, 0.0),
    ).astype(np.float32)

    feats = np.stack(
        [
            rel_t.astype(np.float32),
            delta_t.astype(np.float32),
            pkt_len.astype(np.float32),
            direction,
        ],
        axis=1,
    )  # [num_packets, 4]

    n = min(len(feats), max_packets)
    X[:n] = feats[:n]
    mask[:n] = 1.0

    return torch.tensor(X), torch.tensor(mask)


def infer_label_from_parent(pcap_path: Path, input_root: Path) -> str:
    """
    Uses the immediate parent folder as the class label if the pcap is in a subdirectory.
    If the pcap is directly in input_root, label becomes 'unknown'.
    """
    if pcap_path.parent == input_root:
        return "unknown"
    return pcap_path.parent.name


def collect_pcaps(input_root: Path):
    pcaps = list(input_root.rglob("*.pcap")) + list(input_root.rglob("*.pcapng"))
    return sorted(pcaps)


def build_dataset(
    input_root: Path,
    target_mac: str,
    max_packets: int,
    data_only: bool,
    show_progress: bool = True,
):
    display_filter = build_display_filter(target_mac, data_only)
    pcap_paths = collect_pcaps(input_root)

    if not pcap_paths:
        raise FileNotFoundError(f"No .pcap or .pcapng files found under {input_root}")

    labels_str = [infer_label_from_parent(p, input_root) for p in pcap_paths]
    class_names = sorted(set(labels_str))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    X_list = []
    mask_list = []
    y_list = []
    metadata = []

    total_pcaps = len(pcap_paths)
    succeeded = 0
    failed = 0

    if show_progress:
        print_progress(0, total_pcaps, succeeded, failed)

    for index, (pcap_path, label_name) in enumerate(zip(pcap_paths, labels_str), start=1):
        try:
            df = run_tshark(pcap_path, display_filter)
            x, mask = df_to_packet_tensor(
                df,
                max_packets=max_packets,
                target_mac=target_mac,
            )

            X_list.append(x)
            mask_list.append(mask)
            y_list.append(label_map[label_name])

            metadata.append({
                "file": str(pcap_path),
                "label_name": label_name,
                "label_id": label_map[label_name],
                "num_packets_after_filter": int(mask.sum().item()),
            })
            succeeded += 1

        except subprocess.CalledProcessError as e:
            failed += 1
            if show_progress:
                print(file=sys.stderr, flush=True)
            print(f"[WARN] tshark failed on {pcap_path}", file=sys.stderr)
            print(e.stderr, file=sys.stderr)

        if show_progress:
            print_progress(index, total_pcaps, succeeded, failed)

    if not X_list:
        raise RuntimeError("No captures were successfully processed.")

    X = torch.stack(X_list)         # [N, max_packets, 4]
    mask = torch.stack(mask_list)   # [N, max_packets]
    y = torch.tensor(y_list, dtype=torch.long)

    return X, mask, y, label_map, metadata


def summarize_packet_counts(metadata, suggest_percentile: float):
    counts = np.array(
        [entry["num_packets_after_filter"] for entry in metadata],
        dtype=np.int64,
    )
    if counts.size == 0:
        raise RuntimeError("Cannot summarize packet counts: no captures were processed.")

    percentile_targets = [50, 75, 90, 95, 99]
    summary = {
        "num_captures": int(counts.size),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "percentiles": {
            str(p): float(np.percentile(counts, p)) for p in percentile_targets
        },
        "suggested_max_packets_percentile": float(suggest_percentile),
        "suggested_max_packets": int(np.ceil(np.percentile(counts, suggest_percentile))),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory containing pcap/pcapng files")
    parser.add_argument("--target_mac", type=str, required=True,
                        help="MAC address to filter on (source or destination)")
    parser.add_argument("--max_packets", type=int, default=3000,
                        help="Pad/truncate each capture to this many packets")
    parser.add_argument("--data_only", action="store_true",
                        help="Keep only data frames (wlan.fc.type == 2)")
    parser.add_argument("--output_prefix", type=str, default="dataset",
                        help="Prefix for saved output files")
    parser.add_argument("--report_lengths", action="store_true",
                        help="Print packet-count statistics after filtering and save them to JSON")
    parser.add_argument("--suggest_percentile", type=float, default=95.0,
                        help="Percentile used to recommend max_packets when --report_lengths is set")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable the pcap conversion progress bar")

    args = parser.parse_args()

    input_root = Path(args.input_root)

    if not 0.0 <= args.suggest_percentile <= 100.0:
        raise ValueError("--suggest_percentile must be between 0 and 100")

    X, mask, y, label_map, metadata = build_dataset(
        input_root=input_root,
        target_mac=args.target_mac.lower(),
        max_packets=args.max_packets,
        data_only=args.data_only,
        show_progress=not args.no_progress,
    )

    out_pt = f"{args.output_prefix}.pt"
    out_json = f"{args.output_prefix}_meta.json"

    torch.save({
        "X": X,               # [N, max_packets, 4]
        "mask": mask,         # [N, max_packets]
        "y": y,               # [N]
        "label_map": label_map,
        "feature_names": [
            "relative_time",
            "delta_time",
            "packet_length",
            "direction",
        ],
    }, out_pt)

    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    if args.report_lengths:
        length_summary = summarize_packet_counts(
            metadata=metadata,
            suggest_percentile=args.suggest_percentile,
        )
        out_lengths = f"{args.output_prefix}_lengths.json"
        with open(out_lengths, "w") as f:
            json.dump(length_summary, f, indent=2)
        print(f"Saved length summary to: {out_lengths}")
        print("Packet count summary after filtering:")
        print(json.dumps(length_summary, indent=2))

    print(f"Saved tensor dataset to: {out_pt}")
    print(f"Saved metadata to:       {out_json}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"mask shape: {tuple(mask.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"Classes: {label_map}")


if __name__ == "__main__":
    main()
