#!/bin/bash
# T0 single-sample pipeline: metadata -> voxelize/encode -> render -> proxy -> verify
# Usage: bash scripts/skylines/run_t0_single.sh [sha256] [renderer]
#   sha256   default: ddb33c1bdba1999ac314153ea8990cbd_obj_0
#   renderer default: auto (blender first, nvdiffrast fallback)
set -e

SHA256=${1:-ddb33c1bdba1999ac314153ea8990cbd_obj_0}
RENDERER=${2:-auto}

source /data5/jianxin/anaconda3/bin/activate trellis2
cd /data5/jianxin/TRELLIS.2
export PYTHONPATH=/data5/jianxin/TRELLIS.2:$PYTHONPATH

# pick the GPU with the most free memory
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -rn | head -1 | cut -d',' -f1)
fi
echo "[run_t0] sha256=$SHA256 renderer=$RENDERER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python data_toolkit/skylines/build_metadata.py      --instances "$SHA256"
python data_toolkit/skylines/voxelize_and_encode.py --instances "$SHA256"
python data_toolkit/skylines/render_roof.py         --instances "$SHA256" --renderer "$RENDERER"
python data_toolkit/skylines/build_proxy.py         --instances "$SHA256"
python data_toolkit/skylines/verify_t0.py           --instances "$SHA256"

echo "[run_t0] T0 pipeline finished for $SHA256"
