#!/usr/bin/env python3
"""Evaluate local Dopamine TF DQN checkpoints under AtariBench timing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dopamine_games import DEFAULT_EVAL_GAMES, normalize_game


SCRIPT_DIR = Path(__file__).resolve().parent
DOPAMINE_ROOT = SCRIPT_DIR.parent
RL_BASELINES_ROOT = DOPAMINE_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        "--train-root",
        dest="checkpoint_root",
        type=Path,
        default=RL_BASELINES_ROOT / "checkpoints/dopamine/train_dir/dqn",
    )
    parser.add_argument("--result-root", type=Path, default=RL_BASELINES_ROOT / "results/dopamine_tf_dqn")
    parser.add_argument("--games", nargs="*", default=list(DEFAULT_EVAL_GAMES))
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--skip-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-heatmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=SCRIPT_DIR / "result_heatmap/plot_heatmap_raw_with_dopamine.png",
    )
    return parser.parse_args()


def checkpoint_dir(args: argparse.Namespace, slug: str) -> Path:
    return args.checkpoint_root / slug / f"seed_{args.seed}" / "checkpoints"


def has_checkpoint(path: Path) -> bool:
    return path.exists() and (
        any(path.glob("sentinel_checkpoint_complete.*")) or any(path.glob("tf_ckpt-*.index"))
    )


def run_game(args: argparse.Namespace, game: str) -> dict[str, object] | None:
    spec = normalize_game(game)
    ckpt_dir = checkpoint_dir(args, spec.slug)
    if not has_checkpoint(ckpt_dir):
        message = f"missing checkpoint for {spec.slug}: {ckpt_dir}"
        if args.skip_missing:
            print(message)
            return None
        raise FileNotFoundError(message)

    out_dir = args.result_root / spec.slug
    output_json = out_dir / f"{spec.slug}_tf_dqn_{args.num_runs}run_30s.json"
    output_csv = out_dir / f"{spec.slug}_tf_dqn_{args.num_runs}run_30s.csv"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_tf_dqn_30s.py"),
        f"--game={spec.slug}",
        f"--checkpoint-dir={ckpt_dir}",
        "--checkpoint-iteration=latest",
        "--no-legacy-checkpoint",
        f"--num-runs={args.num_runs}",
        f"--seed-start={args.seed_start}",
        f"--output-json={output_json}",
        f"--output-csv={output_csv}",
    ]
    print(f"Running {spec.slug} ({spec.dopamine_name})")
    subprocess.run(cmd, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def args_games(args: argparse.Namespace) -> list[str]:
    return [normalize_game(game).slug for game in args.games]


def run_all(args: argparse.Namespace) -> dict[str, object]:
    summaries = {}
    missing = []
    for game in args_games(args):
        summary = run_game(args, game)
        if summary is None:
            missing.append(game)
        else:
            summaries[game] = summary

    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "agent": "dopamine_tf_dqn",
        "available_games": sorted(summaries),
        "games_requested": args_games(args),
        "missing_games": missing,
        "num_runs_per_game": args.num_runs,
        "raw_frame_budget": 900,
        "frames_per_action": 3,
        "result_root": str(args.result_root),
        "checkpoint_root": str(args.checkpoint_root),
    }
    manifest_path = args.result_root / "dopamine_tf_dqn_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)
    if args.plot_heatmap:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "result_heatmap/plot_heatmap.py"),
                f"--dopamine-result-root={args.result_root}",
                f"--output={args.heatmap_output}",
            ],
            check=True,
        )
    return manifest


def main() -> int:
    args = parse_args()
    run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
