#!/usr/bin/env python3
"""Build a standalone AtariBench raw-score heatmap with Dopamine results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RL_BASELINES_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_BASELINE_CSV = SCRIPT_DIR / "baseline_scores.csv"
DEFAULT_DOPAMINE_RESULT_ROOT = RL_BASELINES_ROOT / "results/dopamine_tf_dqn"
DEFAULT_OUTPUT = SCRIPT_DIR / "plot_heatmap_raw_with_dopamine.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--dopamine-result-root", type=Path, default=DEFAULT_DOPAMINE_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dopamine-label", default="Dopamine DQN")
    return parser.parse_args()


def load_baseline_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "row_key": row["row_key"],
                    "label": row["label"],
                    "game": row["game"],
                    "score": float(row["score"]),
                    "run_count": int(float(row["run_count"])) if row.get("run_count") else None,
                }
            )
    return rows


def load_dopamine_rows(result_root: Path, label: str) -> list[dict[str, object]]:
    rows_by_game: dict[str, dict[str, object]] = {}
    for path in sorted(result_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("agent") != "dopamine_tf_dqn":
            continue
        if payload.get("checkpoint_kind") == "random_init":
            continue
        game = payload.get("game")
        score = payload.get("avg_total_reward_30s")
        if game is None or score is None:
            continue
        candidate = {
            "row_key": "dopamine_tf_dqn",
            "label": label,
            "game": str(game),
            "score": float(score),
            "run_count": int(payload.get("num_runs", 0) or 0),
            "source": str(path),
        }
        old = rows_by_game.get(str(game))
        if old is None or candidate["score"] > old["score"]:
            rows_by_game[str(game)] = candidate
    return [rows_by_game[game] for game in sorted(rows_by_game)]


def compute_rank_matrix(matrix: np.ndarray) -> np.ndarray:
    n_models, n_games = matrix.shape
    rank_matrix = np.full((n_models, n_games), np.nan)
    for j in range(n_games):
        rows_with_data = [i for i in range(n_models) if not np.isnan(matrix[i, j])]
        col = matrix[rows_with_data, j]
        sorted_order = np.argsort(-col)
        rank = 1
        for k, ki in enumerate(sorted_order):
            idx = rows_with_data[ki]
            if k > 0 and col[ki] == col[sorted_order[k - 1]]:
                prev_idx = rows_with_data[sorted_order[k - 1]]
                rank_matrix[idx, j] = rank_matrix[prev_idx, j]
            else:
                rank_matrix[idx, j] = rank
            rank = k + 2
    return rank_matrix


def normalized_by_game(matrix: np.ndarray) -> np.ndarray:
    n_models, n_games = matrix.shape
    norm = np.full((n_models, n_games), np.nan)
    for j in range(n_games):
        col = matrix[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        col_min = col[valid].min()
        shift = max(0.0, -col_min)
        shifted = col + shift
        col_max = max(shifted[valid].max(), 1e-9)
        norm[valid, j] = np.clip(shifted[valid] / col_max, 0.0, 1.0)
    return norm


def build_matrix(rows: list[dict[str, object]]) -> tuple[list[str], list[str], dict[str, str], np.ndarray]:
    games = []
    for row in rows:
        game = str(row["game"])
        if game not in games:
            games.append(game)

    # Preserve the frozen baseline game order, but do not add non-baseline games
    # without complete comparison rows.
    baseline_games = [str(row["game"]) for row in rows if row["row_key"] != "dopamine_tf_dqn"]
    games = []
    for game in baseline_games:
        if game not in games:
            games.append(game)

    row_keys = []
    labels = {}
    for row in rows:
        row_key = str(row["row_key"])
        if row_key not in row_keys:
            row_keys.append(row_key)
        labels[row_key] = str(row["label"])

    reward = {(str(row["row_key"]), str(row["game"])): float(row["score"]) for row in rows}
    matrix = np.full((len(row_keys), len(games)), np.nan)
    for i, row_key in enumerate(row_keys):
        for j, game in enumerate(games):
            if (row_key, game) in reward:
                matrix[i, j] = reward[(row_key, game)]
    return row_keys, games, labels, matrix


def sort_rows(row_keys: list[str], matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    rank = compute_rank_matrix(matrix)
    norm = normalized_by_game(matrix)
    avg_rank = np.nanmean(rank, axis=1)
    avg_norm = np.nanmean(norm, axis=1)
    order = sorted(range(len(row_keys)), key=lambda i: (avg_rank[i], -avg_norm[i], row_keys[i]))
    return [row_keys[i] for i in order], matrix[order, :]


def plot_raw_heatmap(
    row_keys: list[str],
    games: list[str],
    labels: dict[str, str],
    matrix: np.ndarray,
    output: Path,
) -> None:
    norm = normalized_by_game(matrix)
    cmap_raw = mcolors.LinearSegmentedColormap.from_list(
        "score_balanced",
        ["#d77f7a", "#f3e6bd", "#4daf6c"],
    )
    grey_color = "#cccccc"
    fig_width = max(8.0, len(games) * 0.42)
    fig_height = max(3.6, len(row_keys) * 0.26 + 0.95)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor(grey_color)
    im = ax.imshow(norm, aspect="auto", cmap=cmap_raw, vmin=0, vmax=1, interpolation="nearest")

    for i in range(len(row_keys)):
        for j in range(len(games)):
            if np.isnan(norm[i, j]):
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2))
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="#888888", fontstyle="italic", zorder=3)
            else:
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#222222",
                    fontweight="bold",
                    zorder=3,
                )

    ax.set_xticks(range(len(games)))
    ax.set_xticklabels([game.replace("_", " ").title() for game in games], rotation=40, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([labels[row_key] for row_key in row_keys], fontsize=7)
    ax.set_title("Individual Benchmark Scores by Model (Raw)", fontsize=9.5, fontweight="bold", pad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Normalized Score (per game)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(output)


def main() -> int:
    args = parse_args()
    rows = load_baseline_rows(args.baseline_csv)
    dopamine_rows = load_dopamine_rows(args.dopamine_result_root, args.dopamine_label)
    if dopamine_rows:
        rows.extend(dopamine_rows)
    else:
        print(f"No Dopamine eval summaries found under {args.dopamine_result_root}")

    row_keys, games, labels, matrix = build_matrix(rows)
    row_keys, matrix = sort_rows(row_keys, matrix)
    plot_raw_heatmap(row_keys, games, labels, matrix, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
