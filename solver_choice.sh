# run solver_choice.py
# use ode_solver rk4 & euler

mkdir -p solver_choice_rk4log

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method dflow \
  --init random \
  --ode_solver rk4 \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_rk4/dflow_random \
  > solver_choice_rk4log/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method unrolled \
  --init random \
  --ode_solver rk4 \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_rk4/unrolled_random \
  > solver_choice_rk4log/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method unrolled \
  --init dflow \
  --ode_solver rk4 \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_rk4/unrolled_dflow \
  > solver_choice_rk4log/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method adjoint \
  --init dflow \
  --ode_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_rk4/adjoint_dflow \
  > solver_choice_rk4log/adjoint_dflow.log 2>&1





mkdir -p solver_choice_eulerlog

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method dflow \
  --init random \
  --ode_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_euler/dflow_random \
  > solver_choice_eulerlog/dflow_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method unrolled \
  --init random \
  --ode_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_euler/unrolled_random \
  > solver_choice_eulerlog/unrolled_random.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method unrolled \
  --init dflow \
  --ode_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_euler/unrolled_dflow \
  > solver_choice_eulerlog/unrolled_dflow.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 solver_choice.py \
  --method adjoint \
  --init dflow \
  --ode_solver euler \
  --seeds 0,1,42,123,3407 \
  --iter_max 600 \
  --device cuda \
  --output_dir solver_choice_euler/adjoint_dflow \
  > solver_choice_eulerlog/adjoint_dflow.log 2>&1