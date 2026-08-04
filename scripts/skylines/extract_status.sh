#!/bin/bash
# Summarize progress and failures of a running/finished extraction.
# Usage: bash scripts/skylines/extract_status.sh
LOG_DIR=/data5/jianxin/TRELLIS.2/outputs/extract_logs

echo "===== shard progress ($(date '+%F %T')) ====="
for f in "$LOG_DIR"/progress_rank*.json; do
    [ -e "$f" ] || { echo "no progress files in $LOG_DIR"; break; }
    python3 -c "
import json, sys
p = json.load(open('$f'))
eta_h = p['eta_sec'] / 3600
alive = 'RUNNING' if p['processed'] < p['shard_total'] else 'FINISHED'
print(f\"rank {p['rank']}: {p['processed']}/{p['shard_total']} \"
      f\"done={p['done']} failed={p['failed']} \"
      f\"{p['sec_per_instance']}s/inst eta={eta_h:.1f}h \"
      f\"stages={p['stage_avg_sec']} [{alive}] @{p['updated_at']}\")
"
done

echo ""
echo "===== live processes ====="
for f in "$LOG_DIR"/shard_*.pid; do
    [ -e "$f" ] || break
    PID=$(cat "$f")
    RANK=$(basename "$f" .pid | sed 's/shard_//')
    if kill -0 "$PID" 2>/dev/null; then
        echo "rank $RANK: pid $PID alive"
    else
        echo "rank $RANK: pid $PID exited"
    fi
done

echo ""
echo "===== failures ====="
TOTAL=0
for f in "$LOG_DIR"/failures_rank*.csv; do
    [ -e "$f" ] || { echo "none"; break; }
    N=$(( $(wc -l < "$f") - 1 ))
    TOTAL=$((TOTAL + N))
    echo "$(basename "$f"): $N"
done
[ "$TOTAL" -gt 0 ] && { echo "--- by stage ---"; \
    tail -q -n +2 "$LOG_DIR"/failures_rank*.csv | cut -d',' -f2 | sort | uniq -c; }

echo ""
echo "===== artifact counts ====="
ROOT=/data5/jianxin/dataset/skylines_50k_data_extracted
echo "shape latents: $(ls "$ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_512" 2>/dev/null | grep -c npz)"
echo "render_cond:   $(find "$ROOT/render_cond" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo "proxy:         $(ls "$ROOT/proxy" 2>/dev/null | grep -c npz)"
