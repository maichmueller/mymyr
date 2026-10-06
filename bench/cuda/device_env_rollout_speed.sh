#!/usr/bin/env bash
# Random-policy rollouts of the device environments (mymyr_device_env_rollout_bench) on four reference tasks, one
# JSON line per (task, N).
#
#   bench/cuda/device_env_rollout_speed.sh [build dir, default build/cuda] [extra mymyr_device_env_rollout_bench arguments, e.g. --witness]
#
# Run from the repository root with CUDA_VISIBLE_DEVICES set to the GPU to use.
set -u
BUILD=${1:-build/cuda}
[ $# -gt 0 ] && shift
BIN=$BUILD/cuda/mymyr_device_env_rollout_bench
T=tests/data/tasks
TASKS="miconic__s7-4 blocks__probBLOCKS-8-0 gripper__prob05 sokoban-opt08-strips__p14"
for t in $TASKS; do
  "$BIN" "$T/$t.txt" "$@"
done
uptime
