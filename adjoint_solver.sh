# run adjoint_solver.py
# use reverse_solver rk4 & euler


mkdir -p adjoint_solverrk4log

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method dflow \
  --init random \
  --reverse_solver rk4 \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solverrk4/dflow_random \
  > adjoint_solverrk4log/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method unrolled \
  --init random \
  --reverse_solver rk4 \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solverrk4/unrolled_random \
  > adjoint_solverrk4log/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method unrolled \
  --init dflow \
  --reverse_solver rk4 \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solverrk4/unrolled_dflow \
  > adjoint_solverrk4log/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method adjoint \
  --init dflow \
  --reverse_solver rk4 \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solverrk4/adjoint_dflow \
  > adjoint_solverrk4log/adjoint_dflow.log 2>&1




mkdir -p adjoint_solvereulerlog

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method dflow \
  --init random \
  --reverse_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solvereuler/dflow_random \
  > adjoint_solvereulerlog/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method unrolled \
  --init random \
  --reverse_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solvereuler/unrolled_random \
  > adjoint_solvereulerlog/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method unrolled \
  --init dflow \
  --reverse_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solvereuler/unrolled_dflow \
  > adjoint_solvereulerlog/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 adjoint_solver.py \
  --method adjoint \
  --init dflow \
  --reverse_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir adjoint_solvereuler/adjoint_dflow \
  > adjoint_solvereulerlog/adjoint_dflow.log 2>&1