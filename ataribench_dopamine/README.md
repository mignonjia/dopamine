# AtariBench Dopamine Eval

This folder only evaluates Dopamine TensorFlow DQN checkpoints under the
AtariBench 30-second protocol.

Run everything from the Dopamine repo root:

```bash
cd /Users/mingjiahuo/Desktop/ataribench/rl_baselines/dopamine
source ~/.zshrc
conda activate ale
```

## Single Checkpoint

Evaluate one checkpoint by TensorFlow checkpoint prefix:

```bash
python ataribench_dopamine/evaluate_tf_dqn_30s.py \
  --game breakout \
  --checkpoint-prefix ../checkpoints/dopamine/dqn/Breakout/1/tf_ckpt-199 \
  --num-runs 10
```

For checkpoints saved by a current Dopamine training run, pass the checkpoint
directory instead:

```bash
python ataribench_dopamine/evaluate_tf_dqn_30s.py \
  --game breakout \
  --checkpoint-dir ../checkpoints/dopamine/train_dir/dqn/breakout/seed_1111/checkpoints \
  --no-legacy-checkpoint \
  --num-runs 10
```

Outputs are written to:

```text
../results/dopamine_tf_dqn/<game>/<game>_tf_dqn_<N>run_30s.json
../results/dopamine_tf_dqn/<game>/<game>_tf_dqn_<N>run_30s.csv
```

## Batch Eval

Batch eval assumes checkpoints live at:

```text
../checkpoints/dopamine/train_dir/dqn/<game>/seed_1111/checkpoints/
```

Run the default eval set with available checkpoints:

```bash
python ataribench_dopamine/evaluate_all_30s.py \
  --checkpoint-root ../checkpoints/dopamine/train_dir/dqn \
  --result-root ../results/dopamine_tf_dqn \
  --seed 1111 \
  --num-runs 10
```

Run a subset:

```bash
python ataribench_dopamine/evaluate_all_30s.py \
  --games breakout assault qbert \
  --num-runs 10
```

Missing checkpoints are skipped by default. Use `--no-skip-missing` to fail
instead.

## Download Legacy Checkpoint

The old Dopamine/Lucid checkpoints are stored as three TensorFlow files:

```text
tf_ckpt-199.index
tf_ckpt-199.meta
tf_ckpt-199.data-00000-of-00001
```

Download one with:

```bash
GAME=Breakout RUN=1 AGENT=dqn CHECKPOINT=tf_ckpt-199 \
  bash ataribench_dopamine/download_tf_dqn_ckpt.sh
```

Then run the single-checkpoint command above.

## Eval Protocol

- `ALE/<Game>-v5`
- `frameskip=1`
- `repeat_action_probability=0.0`
- reduced action space
- `900` raw frames per run
- `3` raw frames per selected DQN action
- raw ALE reward, not clipped reward

`ATARIBENCH_GAMES` / default eval slugs:

```text
air_raid assault beam_rider boxing breakout demon_attack fishing_derby freeway
gopher ice_hockey journey_escape name_this_game pacman phoenix qbert riverraid
robotank seaquest tennis time_pilot
```

Explicit-only slugs:

```text
laser_gates
```
