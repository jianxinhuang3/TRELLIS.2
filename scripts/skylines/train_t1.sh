#!/bin/bash
# T1: Train SS flow injection layers (single-sample overfit test)
set -e
source /data5/jianxin/anaconda3/bin/activate trellis2
cd /data5/jianxin/TRELLIS.2
export PYTHONPATH=/data5/jianxin/TRELLIS.2:$PYTHONPATH

# Pick GPU with most free memory
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -rn | head -1 | cut -d',' -f1)
fi
echo "[train_t1] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

DATA_DIR='{"skylines": {"metadata": "/data5/jianxin/dataset/skylines_50k_data_extracted", "ss_latent": "/data5/jianxin/dataset/skylines_50k_data_extracted/ss_latents/ss_enc_conv3d_16l8_fp16_64", "render_cond": "/data5/jianxin/dataset/skylines_50k_data_extracted/render_cond", "proxy": "/data5/jianxin/dataset/skylines_50k_data_extracted/proxy"}}'

python train.py \
  --config configs/skylines/t1_ss_flow_inject.json \
  --output_dir outputs/skylines_t1_toy \
  --data_dir "$DATA_DIR" \
  --num_gpus 1
