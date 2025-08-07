# run 4methods.py & 4methods_5imageset.py

mkdir -p 4methods_resultslog
mkdir -p 4methods_5imagesetlog

CUDA_VISIBLE_DEVICES=1 python3 4methods.py \
  --method unrolled \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir 4methods_results/unrolled_random \
  > 4methods_resultslog/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 4methods.py \
  --method unrolled \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir 4methods_results/unrolled_dflow \
  > 4methods_resultslog/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 4methods.py \
  --method adjoint \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir 4methods_results/adjoint_dflow \
  > 4methods_resultslog/adjoint_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 4methods.py \
  --method dflow \
  --init random \
  --seeds 0,1,42,123,3407 \
  --iter_max 1000 \
  --device cuda \
  --output_dir 4methods_results/dflow_random \
  > 4methods_resultslog/dflow_random.log 2>&1


CUDA_VISIBLE_DEVICES=1 python3 4methods_5imageset.py \
  --method dflow \
  --init random \
  --seeds 0,1,42,123,3407 \
  --num_images 5 \
  --iter_max 600 \
  --device cuda \
  --output_dir 4methods_5imageset/dflow_random \
  > 4methods_5imagesetlog/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 4methods_5imageset.py \
  --method unrolled \
  --init random \
  --seeds 0,1,42,123,3407 \
  --num_images 5 \
  --iter_max 600 \
  --device cuda \
  --output_dir 4methods_5imageset/unrolled_random \
  > 4methods_5imagesetlog/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 4methods_5imageset.py \
  --method unrolled \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --num_images 5 \
  --iter_max 600 \
  --device cuda \
  --output_dir 4methods_5imageset/unrolled_dflow \
  > 4methods_5imagesetlog/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 4methods_5imageset.py \
  --method adjoint \
  --init dflow \
  --seeds 0,1,42,123,3407 \
  --num_images 5 \
  --iter_max 600 \
  --device cuda \
  --output_dir 4methods_5imageset/adjoint_dflow \
  > 4methods_5imagesetlog/adjoint_dflow.log 2>&1
