#!/usr/bin/env python3
"""Evaluate a Dopamine TensorFlow DQN checkpoint under AtariBench timing."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import random
import re
import sys
import types
from collections import namedtuple
from pathlib import Path

import numpy as np

from dopamine_games import normalize_game


SCRIPT_DIR = Path(__file__).resolve().parent
DOPAMINE_ROOT = SCRIPT_DIR.parent
RL_BASELINES_ROOT = DOPAMINE_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="breakout")
    parser.add_argument(
        "--checkpoint-prefix",
        type=Path,
        help="TensorFlow checkpoint prefix, e.g. path/to/tf_ckpt-199.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Dopamine train checkpoint directory containing tf_ckpt-* and sentinels.",
    )
    parser.add_argument("--checkpoint-iteration", default="latest")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--raw-frame-budget", type=int, default=900)
    parser.add_argument("--frames-per-action", type=int, default=3)
    parser.add_argument("--eval-epsilon", type=float, default=0.001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RL_BASELINES_ROOT / "results/dopamine_tf_dqn",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--legacy-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def checkpoint_file(prefix: Path, suffix: str) -> Path:
    return Path(str(prefix) + suffix)


def latest_checkpoint_prefix(checkpoint_dir: Path) -> Path:
    sentinels = sorted(checkpoint_dir.glob("sentinel_checkpoint_complete.*"))
    if sentinels:
        latest = max(int(path.name.rsplit(".", 1)[1]) for path in sentinels)
        return checkpoint_dir / f"tf_ckpt-{latest}"

    checkpoint_state = checkpoint_dir / "checkpoint"
    if checkpoint_state.exists():
        match = re.search(r'model_checkpoint_path:\s+"([^"]+)"', checkpoint_state.read_text())
        if match:
            ckpt = Path(match.group(1))
            return ckpt if ckpt.is_absolute() else checkpoint_dir / ckpt

    candidates = sorted(checkpoint_dir.glob("tf_ckpt-*.index"))
    if not candidates:
        raise FileNotFoundError(f"No tf_ckpt-*.index files found in {checkpoint_dir}")
    latest = max(int(path.stem.rsplit("-", 1)[1]) for path in candidates)
    return checkpoint_dir / f"tf_ckpt-{latest}"


def resolve_checkpoint_prefix(args: argparse.Namespace) -> Path:
    if args.checkpoint_prefix is not None:
        return args.checkpoint_prefix
    if args.checkpoint_dir is not None:
        if args.checkpoint_iteration == "latest":
            return latest_checkpoint_prefix(args.checkpoint_dir)
        return args.checkpoint_dir / f"tf_ckpt-{int(args.checkpoint_iteration)}"
    spec = normalize_game(args.game)
    return RL_BASELINES_ROOT / "checkpoints/dopamine/dqn" / spec.dopamine_name / "1" / "tf_ckpt-199"


def install_atari_lib_stub() -> None:
    """Avoid importing old Baselines just to get Dopamine network namedtuples."""

    module = types.ModuleType("dopamine.discrete_domains.atari_lib")
    module.DQNNetworkType = namedtuple("dqn_network", ["q_values"])
    module.RainbowNetworkType = namedtuple(
        "c51_network", ["q_values", "logits", "probabilities"]
    )
    module.ImplicitQuantileNetworkType = namedtuple(
        "iqn_network", ["quantile_values", "quantiles"]
    )
    sys.modules["dopamine.discrete_domains.atari_lib"] = module


class AtariBenchPreprocessor:
    """Minimal Dopamine-style visual preprocessing for one-frame observations."""

    def __init__(self, screen_size: int = 84):
        self.screen_size = screen_size
        self._last_gray: np.ndarray | None = None

    def reset(self, rgb: np.ndarray) -> np.ndarray:
        gray = self._gray(rgb)
        self._last_gray = gray
        return self._resize(gray)

    def process(self, rgb: np.ndarray) -> np.ndarray:
        gray = self._gray(rgb)
        if self._last_gray is None:
            pooled = gray
        else:
            pooled = np.maximum(gray, self._last_gray)
        self._last_gray = gray
        return self._resize(pooled)

    def _gray(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _resize(self, gray: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(
            gray,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.uint8)


def build_env(game: str, render_mode: str | None = None):
    import ale_py
    import gymnasium as gym

    spec = normalize_game(game)
    gym.register_envs(ale_py)
    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(
        f"ALE/{spec.dopamine_name}-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        **kwargs,
    )


def build_agent(args: argparse.Namespace, num_actions: int):
    sys.path.insert(0, str(DOPAMINE_ROOT))
    install_atari_lib_stub()

    import tensorflow as tf
    from dopamine.discrete_domains import legacy_networks
    from dopamine.tf.agents.dqn import dqn_agent

    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.reset_default_graph()
    random.seed(args.seed_start)
    np.random.seed(args.seed_start)

    agent = dqn_agent.DQNAgent(
        sess=None,
        num_actions=num_actions,
        eval_mode=True,
        epsilon_eval=args.eval_epsilon,
        min_replay_history=1,
        tf_device="/cpu:*",
        summary_writer=None,
    )
    if args.legacy_checkpoint:
        var_map = legacy_networks.maybe_transform_variable_names(
            tf.compat.v1.global_variables(),
            legacy_checkpoint_load=True,
        )
    else:
        var_map = None
    saver = tf.compat.v1.train.Saver(var_list=var_map)
    saver.restore(agent._sess, str(args.checkpoint_prefix))  # pylint: disable=protected-access
    agent.eval_mode = True
    return agent


def reset_env(env, seed: int | None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action: int):
    result = env.step(int(action))
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, float(reward), bool(done), info


def action_meanings(env) -> list[str]:
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "get_action_meanings"):
        return list(unwrapped.get_action_meanings())
    return []


def capture_render_frame(env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, list):
        frame = frame[0]
    return np.asarray(frame)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            rgb = np.asarray(frame)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def evaluate_one_run(args: argparse.Namespace, agent, run_index: int) -> dict[str, object]:
    seed = args.seed_start + run_index
    random.seed(seed)
    np.random.seed(seed)

    spec = normalize_game(args.game)
    record_video = args.output_video is not None and run_index == 0
    env = build_env(args.game, render_mode="rgb_array" if record_video else None)
    preprocessor = AtariBenchPreprocessor()
    frames: list[np.ndarray] = []
    try:
        obs = reset_env(env, seed)
        if record_video:
            frames.append(capture_render_frame(env))
        processed = preprocessor.reset(obs)
        action = agent.begin_episode(processed)
        raw_frames = 0
        total_reward = 0.0
        episode_count = 1
        terminal_count = 0
        decisions = 0
        action_counts: Counter[int] = Counter()
        meanings = action_meanings(env)

        while raw_frames < args.raw_frame_budget:
            action_reward = 0.0
            done = False
            processed = None
            action_id = int(action)
            action_counts[action_id] += 1
            for _ in range(args.frames_per_action):
                if raw_frames >= args.raw_frame_budget:
                    break
                obs, reward, done, _info = step_env(env, action_id)
                raw_frames += 1
                action_reward += reward
                total_reward += reward
                if record_video:
                    frames.append(capture_render_frame(env))
                processed = preprocessor.process(obs)
                if done:
                    break
            decisions += 1
            if done:
                terminal_count += 1
                agent.end_episode(action_reward)
                if raw_frames >= args.raw_frame_budget:
                    break
                obs = reset_env(env, None)
                processed = preprocessor.reset(obs)
                action = agent.begin_episode(processed)
                episode_count += 1
            else:
                assert processed is not None
                action = agent.step(action_reward, processed)

        row = {
            "run_index": run_index + 1,
            "seed": seed,
            "game": spec.slug,
            "dopamine_game": spec.dopamine_name,
            "agent": "dopamine_tf_dqn",
            "checkpoint_prefix": str(args.checkpoint_prefix),
            "raw_frame_budget": args.raw_frame_budget,
            "raw_frames_executed": raw_frames,
            "frames_per_action": args.frames_per_action,
            "observed_frames_including_reset": raw_frames + 1,
            "fire_reset": False,
            "sticky_actions": False,
            "ale_frameskip": 1,
            "action_meanings": meanings,
            "action_counts": dict(sorted(action_counts.items())),
            "total_reward": total_reward,
            "decision_count": decisions,
            "episode_count": episode_count,
            "terminal_count": terminal_count,
            "episode_reset_count": max(0, episode_count - 1),
        }
        if record_video:
            frames = frames[: args.raw_frame_budget + 1]
            write_video(args.output_video, frames, args.video_fps)
            row["video_path"] = str(args.output_video)
            row["video_frames"] = len(frames)
        return row
    finally:
        env.close()


def write_outputs(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    spec = normalize_game(args.game)
    output_dir = args.output_dir / spec.slug if args.output_json is None and args.output_csv is None else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.slug}_tf_dqn_{len(rows)}run_30s"
    json_path = args.output_json or output_dir / f"{stem}.json"
    csv_path = args.output_csv or output_dir / f"{stem}.csv"
    summary = {
        "game": spec.slug,
        "dopamine_game": spec.dopamine_name,
        "agent": "dopamine_tf_dqn",
        "checkpoint_prefix": str(args.checkpoint_prefix),
        "checkpoint_kind": "legacy_tf" if args.legacy_checkpoint else "dopamine_tf",
        "num_runs": len(rows),
        "avg_total_reward_30s": float(np.mean([row["total_reward"] for row in rows])),
        "stderr_total_reward_30s": float(
            np.std([row["total_reward"] for row in rows], ddof=1) / np.sqrt(len(rows))
        )
        if len(rows) > 1
        else 0.0,
        "best_total_reward_30s": float(max(row["total_reward"] for row in rows)),
        "worst_total_reward_30s": float(min(row["total_reward"] for row in rows)),
        "raw_frame_budget": args.raw_frame_budget,
        "frames_per_action": args.frames_per_action,
        "eval_epsilon": args.eval_epsilon,
        "runs": rows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)
    print(json.dumps({k: summary[k] for k in ("avg_total_reward_30s", "stderr_total_reward_30s")}))


def main() -> int:
    args = parse_args()
    args.checkpoint_prefix = resolve_checkpoint_prefix(args)
    required = [
        checkpoint_file(args.checkpoint_prefix, ".index"),
        checkpoint_file(args.checkpoint_prefix, ".meta"),
        checkpoint_file(args.checkpoint_prefix, ".data-00000-of-00001"),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint files: " + ", ".join(str(path) for path in missing)
        )

    env = build_env(args.game)
    try:
        num_actions = env.action_space.n
    finally:
        env.close()
    agent = build_agent(args, num_actions)
    rows = [evaluate_one_run(args, agent, run_index) for run_index in range(args.num_runs)]
    write_outputs(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
