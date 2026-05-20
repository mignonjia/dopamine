#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOPAMINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RL_BASELINES_ROOT="$(cd "${DOPAMINE_ROOT}/.." && pwd)"
AGENT="${AGENT:-dqn}"
GAME="${GAME:-Breakout}"
RUN="${RUN:-1}"
CHECKPOINT="${CHECKPOINT:-tf_ckpt-199}"
OUT_DIR="${OUT_DIR:-${RL_BASELINES_ROOT}/checkpoints/dopamine/${AGENT}/${GAME}/${RUN}}"

mkdir -p "${OUT_DIR}"

for suffix in index meta data-00000-of-00001; do
  src="gs://download-dopamine-rl/lucid/${AGENT}/${GAME}/${RUN}/${CHECKPOINT}.${suffix}"
  echo "Downloading ${src}"
  gsutil cp "${src}" "${OUT_DIR}/"
done

echo "${OUT_DIR}/${CHECKPOINT}"
