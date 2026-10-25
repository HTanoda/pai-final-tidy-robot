#!/usr/bin/env bash
# Franka Panda モデル（MuJoCo Menagerie, Apache-2.0）を third_party/ に取得する
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
fi
echo "ok: third_party/mujoco_menagerie/franka_emika_panda"
