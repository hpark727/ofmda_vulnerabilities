#!/usr/bin/env python3

import argparse
import json
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "radiotap.dbm_antsignal",
    "wlan.sa",
    "wlan.da",
]


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


def df_to_packet_tensor(df: pd.DataFrame, max_packets: int):
    """
    Convert one filtered capture DataFrame into:
      X:    [max_packets, 4]
      mask: [max_packets]
    features = [relative_time, delta_time, packet_length, rssi]
    """
    X = np.zeros((max_packets, 4), dtype=np.float32)
    mask = np.zeros((max_packets,), dtype=np.float32)

    if df.empty:
        return torch.tensor(X), torch.tensor(mask)

    # clean numeric columns
    for col in ["frame.time_epoch", "frame.len", "radiotap.dbm_antsignal"]:
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

    # RSSI: fill missing with per-capture median, else fallback -100
    rssi = df["radiotap.dbm_antsignal"].to_numpy(dtype=np.float32)
    if np.all(np.isnan(rssi)):
        rssi = np.full_like(pkt_len, -100.0, dtype=np.float32)
    else:
        median_rssi = np.nanmedian(rssi)
        rssi = np.where(np.isnan(rssi), median_rssi, rssi).astype(np.float32)

    feats = np.stack(
        [
            rel_t.astype(np.float32),
            delta_t.astype(np.float32),
            pkt_len.astype(np.float32),
            rssi.astype(np.float32),
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


def build_dataset(input_root: Path, target_mac: str, max_packets: int, data_only: bool):
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

    for pcap_path, label_name in zip(pcap_paths, labels_str):
        try:
            df = run_tshark(pcap_path, display_filter)
            x, mask = df_to_packet_tensor(df, max_packets=max_packets)

            X_list.append(x)
            mask_list.append(mask)
            y_list.append(label_map[label_name])

            metadata.append({
                "file": str(pcap_path),
                "label_name": label_name,
                "label_id": label_map[label_name],
                "num_packets_after_filter": int(mask.sum().item()),
            })

        except subprocess.CalledProcessError as e:
            print(f"[WARN] tshark failed on {pcap_path}")
            print(e.stderr)

    if not X_list:
        raise RuntimeError("No captures were successfully processed.")

    X = torch.stack(X_list)         # [N, max_packets, 4]
    mask = torch.stack(mask_list)   # [N, max_packets]
    y = torch.tensor(y_list, dtype=torch.long)

    return X, mask, y, label_map, metadata


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

    args = parser.parse_args()

    input_root = Path(args.input_root)

    X, mask, y, label_map, metadata = build_dataset(
        input_root=input_root,
        target_mac=args.target_mac.lower(),
        max_packets=args.max_packets,
        data_only=args.data_only,
    )

    out_pt = f"{args.output_prefix}.pt"
    out_json = f"{args.output_prefix}_meta.json"

    torch.save({
        "X": X,               # [N, max_packets, 4]
        "mask": mask,         # [N, max_packets]
        "y": y,               # [N]
        "label_map": label_map,
        "feature_names": ["relative_time", "delta_time", "packet_length", "rssi"],
    }, out_pt)

    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved tensor dataset to: {out_pt}")
    print(f"Saved metadata to:       {out_json}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"mask shape: {tuple(mask.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"Classes: {label_map}")


if __name__ == "__main__":
    main()