#!/bin/bash
# Launch the SatSkylines full-scale extraction as sharded background processes.
#
# Usage: bash scripts/skylines/extract_all.sh [GPUS] [PROCS_PER_GPU] [EXTRA_ARGS...]
#   GPUS          comma separated GPU ids, e.g. "0,1,2,3"   (default: "6,7")
#   PROCS_PER_GPU shards per GPU                            (default: 1)
#   EXTRA_ARGS    forwarded to extract_all.py, e.g. --limit 20 --instances file
#
# Examples:
#   bash scripts/skylines/extract_all.sh "0,1,2,3,4,5,6,7" 1        # full run, 8 shards
#   bash scripts/skylines/extract_all.sh "6,7" 1 --limit 10         # smoke test
#
# Monitor: bash scripts/skylines/extract_status.sh
set -e

GPUS=${1:-"6,7"}
PROCS_PER_GPU=${2:-1}
shift $(( $# >= 2 ? 2 : $# )) || true
EXTRA_ARGS=("$@")

source /data5/jianxin/anaconda3/bin/activate trellis2
cd /data5/jianxin/TRELLIS.2
export PYTHONPATH=/data5/jianxin/TRELLIS.2:$PYTHONPATH

LOG_DIR=/data5/jianxin/TRELLIS.2/outputs/extract_logs
mkdir -p "$LOG_DIR"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
WORLD_SIZE=$(( ${#GPU_ARR[@]} * PROCS_PER_GPU ))
echo "[extract_all.sh] gpus=[$GPUS] procs_per_gpu=$PROCS_PER_GPU world_size=$WORLD_SIZE extra=(${EXTRA_ARGS[*]})"

RANK=0
for GPU in "${GPU_ARR[@]}"; do
    for ((p = 0; p < PROCS_PER_GPU; p++)); do
        LOG="$LOG_DIR/shard_${RANK}.log"
        CUDA_VISIBLE_DEVICES=$GPU nohup python data_toolkit/skylines/extract_all.py \
            --rank "$RANK" --world_size "$WORLD_SIZE" "${EXTRA_ARGS[@]}" \
            > "$LOG" 2>&1 &
        echo "[extract_all.sh] rank=$RANK gpu=$GPU pid=$! log=$LOG"
        echo $! > "$LOG_DIR/shard_${RANK}.pid"
        RANK=$((RANK + 1))
    done
done

echo ""
echo "Monitor:   bash scripts/skylines/extract_status.sh"
echo "Tail log:  tail -f $LOG_DIR/shard_0.log"
echo "Stop all:  cat $LOG_DIR/shard_*.pid | xargs -r kill"
echo "After all shards finish, rebuild metadata:"
echo "           python data_toolkit/skylines/aggregate_metadata.py"
