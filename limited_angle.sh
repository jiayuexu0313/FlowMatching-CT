# run limited_angle.py
# max_angle_deg=90, 60
# num_angles=90, 60

mkdir -p limited90newlog
mkdir -p limited60newlog

CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method dflow \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 90 \
  --num_angles 90 \
  --output_dir limited90new/dflow_random \
  > limited90newlog/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method unrolled \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 90 \
  --num_angles 90 \
  --output_dir limited90new/unrolled_random \
  > limited90newlog/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method unrolled \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 90 \
  --num_angles 90 \
  --output_dir limited90new/unrolled_dflow \
  > limited90newlog/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method adjoint \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 90 \
  --num_angles 90 \
  --output_dir limited90new/adjoint_dflow \
  > limited90newlog/adjoint_dflow.log 2>&1






CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method dflow \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 60 \
  --num_angles 60 \
  --output_dir limited60new/dflow_random \
  > limited60newlog/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method unrolled \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 60 \
  --num_angles 60 \
  --output_dir limited60new/unrolled_random \
  > limited60newlog/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method unrolled \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 60 \
  --num_angles 60 \
  --output_dir limited60new/unrolled_dflow \
  > limited60newlog/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 limited_angle.py \
  --method adjoint \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --max_angle_deg 60 \
  --num_angles 60 \
  --output_dir limited60new/adjoint_dflow \
  > limited60newlog/adjoint_dflow.log 2>&1