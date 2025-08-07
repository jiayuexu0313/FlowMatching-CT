# run sparse.py
# num_angles=30, 60

mkdir -p sparse30log
mkdir -p sparse60log

CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method dflow \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse30/dflow_random \
  --num_angles 30 \
  > sparse30log/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method unrolled \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse30/unrolled_random \
  --num_angles 30 \
  > sparse30log/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method unrolled \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse30/unrolled_dflow \
  --num_angles 30 \
  > sparse30log/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method adjoint \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse30/adjoint_dflow \
  --num_angles 30 \
  > sparse30log/adjoint_dflow.log 2>&1






CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method dflow \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse60/dflow_random \
  --num_angles 60 \
  > sparse60log/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method unrolled \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse60/unrolled_random \
  --num_angles 60 \
  > sparse60log/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method unrolled \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse60/unrolled_dflow \
  --num_angles 60 \
  > sparse60log/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 sparse.py \
  --method adjoint \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir sparse60/adjoint_dflow \
  --num_angles 60 \
  > sparse60log/adjoint_dflow.log 2>&1