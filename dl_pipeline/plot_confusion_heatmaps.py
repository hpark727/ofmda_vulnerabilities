#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def concise_result_name(result_name: str) -> str:
    replacements = {
        "ofdma_on_model_on_ofdma_on_traces": "Train: OFDMA On | Test: OFDMA On",
        "ofdma_off_model_on_ofdma_off_traces": "Train: OFDMA Off | Test: OFDMA Off",
        "ofdma_on_model_on_ofdma_off_traces": "Train: OFDMA On | Test: OFDMA Off",
        "ofdma_off_model_on_ofdma_on_traces": "Train: OFDMA Off | Test: OFDMA On",
    }
    if result_name in replacements:
        return replacements[result_name]
    return result_name.replace("_model_on_", " -> ").replace("_traces", "").replace("_", " ")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )


def select_results(
    data: dict,
    result_names: Optional[Sequence[str]],
    clients: Optional[Sequence[int]],
) -> List[Tuple[dict, dict]]:
    selected = []
    allowed_names = set(result_names or [])
    allowed_clients = set(clients or [])

    for result in data.get("results", []):
        if allowed_names and result.get("name") not in allowed_names:
            continue
        for client_result in result.get("client_results", []):
            if allowed_clients and int(client_result.get("clients")) not in allowed_clients:
                continue
            selected.append((result, client_result))

    return selected


def extract_confusion(client_result: dict, shot: str) -> Tuple[List[str], np.ndarray]:
    shot_result = client_result.get("shots", {}).get(shot)
    if shot_result is None:
        raise KeyError(f"Shot '{shot}' not found for clients={client_result.get('clients')}.")

    report = shot_result["classification_report"]
    confusion = report["confusion_matrix"]
    labels = [entry["label"] for entry in confusion["labels"]]
    matrix = np.array(confusion["rows_true_cols_pred"], dtype=int)
    return labels, matrix


def anonymize_labels(labels: Sequence[str], prefix: str) -> List[str]:
    return [f"{prefix} {idx}" for idx, _ in enumerate(labels, start=1)]


def plot_confusion_heatmap(
    labels: Sequence[str],
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    normalize: bool,
    cmap: str,
    dpi: int,
) -> None:
    sns.set_theme(style="white", context="talk")
    display_matrix = normalize_rows(matrix) if normalize else matrix
    value_format = ".1%" if normalize else "d"

    fig_width = max(8.0, 1.15 * len(labels) + 3.0)
    fig_height = max(6.8, 1.0 * len(labels) + 2.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=display_matrix,
        display_labels=list(labels),
    )
    display.plot(
        ax=ax,
        cmap=sns.color_palette(cmap, as_cmap=True),
        colorbar=True,
        values_format=value_format,
        xticks_rotation=35,
        im_kw={"vmin": 0.0},
    )

    ax.set_title(title, pad=18)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="both", labelsize=10)

    for text in display.text_.ravel():
        text.set_fontsize(10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render sklearn/seaborn confusion-matrix heatmaps from evaluation JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/regular_eval_k5_clients_1_2_4_6_8.json"),
        help="Evaluation JSON containing classification_report confusion matrices.",
    )
    parser.add_argument(
        "--result_names",
        nargs="+",
        default=None,
        help="Optional result names to plot, e.g. ofdma_on_model_on_ofdma_on_traces.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=None,
        help="Optional client counts to plot.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        default=["5_shot"],
        help="Shot keys to plot, e.g. 1_shot 5_shot.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("evaluation/confusion_heatmaps"),
    )
    parser.add_argument(
        "--normalize",
        choices=["true", "false"],
        default="true",
        help="When true, show row-normalized percentages; otherwise show raw counts.",
    )
    parser.add_argument("--cmap", default="Blues")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--anonymize_labels",
        action="store_true",
        help="Display labels as 'site 1', 'site 2', etc. instead of original class names.",
    )
    parser.add_argument("--anonymous_label_prefix", default="site")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    selected = select_results(data, args.result_names, args.clients)
    if not selected:
        raise ValueError("No matching results found in the evaluation JSON.")

    normalize = args.normalize == "true"
    written = []
    for result, client_result in selected:
        result_name = result["name"]
        clients = int(client_result["clients"])
        for shot in args.shots:
            labels, matrix = extract_confusion(client_result, shot)
            if args.anonymize_labels:
                labels = anonymize_labels(labels, args.anonymous_label_prefix)
            suffix = "normalized" if normalize else "counts"
            label_suffix = "_anonymous" if args.anonymize_labels else ""
            output_path = (
                args.output_dir
                / f"{slugify(result_name)}_{clients}clients_{shot}_{suffix}{label_suffix}.png"
            )
            title = (
                f"{concise_result_name(result_name)}\n"
                f"{clients} clients, {shot.replace('_', '-')}"
            )
            plot_confusion_heatmap(
                labels=labels,
                matrix=matrix,
                title=title,
                output_path=output_path,
                normalize=normalize,
                cmap=args.cmap,
                dpi=args.dpi,
            )
            written.append(output_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
