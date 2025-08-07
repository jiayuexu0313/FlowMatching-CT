# run monitor_norm.py

mkdir -p monitor_normlog

CUDA_VISIBLE_DEVICES=1 python3 monitor_norm.py \
  --method dflow \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir monitor_norm/dflow_random \
  > monitor_normlog/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 monitor_norm.py \
  --method unrolled \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir monitor_norm/unrolled_random \
  > monitor_normlog/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 monitor_norm.py \
  --method unrolled \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir monitor_norm/unrolled_dflow \
  > monitor_normlog/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 monitor_norm.py \
  --method adjoint \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir monitor_norm/adjoint_dflow \
  > monitor_normlog/adjoint_dflow.log 2>&1