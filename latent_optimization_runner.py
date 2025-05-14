# ✅ latent_optimization_runner.py (for RK4 and Midpoint support)
import argparse
from flow_latent_opt_new import optimize_discretise
from flow_latent_opt_adjoint_new import optimize_adjoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["discretise", "adjoint"], required=True)
    parser.add_argument("--integrator", default="midpoint", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.mode == "discretise":
        optimize_discretise(integrator=args.integrator, step_size=args.step_size, device=args.device)
    elif args.mode == "adjoint":
        optimize_adjoint(integrator=args.integrator, step_size=args.step_size, device=args.device)

if __name__ == "__main__":
    main()
