import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import ray
import torch

from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray._private import resource_spec


# Monkey patch to skip nvidia-smi check
def _autodetect_num_gpus():
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on the ExploreAgent Gymnasium environment.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-gpus", type=float, default=0)
    parser.add_argument("--env-name", default="rooms", choices=["default", "empty", "level2", "random", "playground", "rooms"])
    parser.add_argument("--reward-mode", default="coverage", choices=["dynamic", "continuous", "static", "coverage"])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--checkpoint-dir", default="tmp/ppo_rooms")
    parser.add_argument("--ray-temp-dir", default=str(Path(tempfile.gettempdir()) / "aiar_ray"))
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    return parser.parse_args()


def get_task_limits(args):
    env = ExploreDrone({
        "env_name": args.env_name,
        "reward_mode": args.reward_mode,
        "max_steps": args.max_steps,
        "gui": False,
    })
    if args.reward_mode == "coverage":
        return {
            "n_checkpoints": env.env.n_goals,
            "checkpoint_reward": env.drone.COVERAGE_REWARD,
            "max_reward": env.env.n_goals * env.drone.COVERAGE_REWARD,
        }
    return {"n_checkpoints": None, "checkpoint_reward": None, "max_reward": None}


def print_training_context(args, task_limits):
    print("\nTraining PPO")
    print(f"  environment: {args.env_name}")
    print(f"  reward mode: {args.reward_mode}")
    print(f"  max steps per episode: {args.max_steps}")
    print(f"  rollout workers: {args.num_workers}")
    print(f"  requested GPUs: {args.num_gpus}")
    print(f"  entropy coefficient: {args.entropy_coeff}")
    if task_limits["max_reward"] is not None:
        print(
            "  coverage task: "
            f"{task_limits['n_checkpoints']} checkpoints x "
            f"{task_limits['checkpoint_reward']:.2f} reward = "
            f"{task_limits['max_reward']:.2f} max reward"
        )
        print("  exploration shaping: small penalty for hovering in already visited map cells")

    print("\nLog columns")
    print("  iter: completed PPO training iteration")
    print("  reward min/mean/max: episode return statistics from the latest training batch")
    if task_limits["max_reward"] is not None:
        print("  score%: mean reward as a percentage of the checkpoint reward maximum")
        print("          idle penalties can reduce returns below the checkpoint score")
        print("          early scores can be negative because collision has a penalty")
    print("  len: mean episode length in environment steps")
    print("  best: best mean reward saved after warmup iterations\n")
    sys.stdout.flush()


def format_training_status(iteration, result, best_reward, max_reward):
    min_reward = result["episode_reward_min"]
    mean_reward = result["episode_reward_mean"]
    max_seen_reward = result["episode_reward_max"]
    mean_len = result["episode_len_mean"]
    best = best_reward if best_reward != float("-inf") else 0.0

    if max_reward is not None and max_reward > 0:
        score_percent = 100.0 * mean_reward / max_reward
        return (
            f"iter {iteration:4d} | "
            f"reward {min_reward:7.2f}/{mean_reward:7.2f}/{max_seen_reward:7.2f} | "
            f"score {score_percent:6.1f}% | "
            f"len {mean_len:6.1f} | "
            f"best {best:7.2f}"
        )

    return (
        f"iter {iteration:4d} | "
        f"reward {min_reward:7.2f}/{mean_reward:7.2f}/{max_seen_reward:7.2f} | "
        f"len {mean_len:6.1f} | "
        f"best {best:7.2f}"
    )


def main():
    args = parse_args()
    if args.num_gpus > 0 and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training was requested with --num-gpus > 0, but this environment has CPU-only PyTorch. "
            "Install a CUDA-enabled PyTorch build or run with --num-gpus 0."
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    task_limits = get_task_limits(args)
    print_training_context(args, task_limits)

    if args.num_gpus == 0:
        resource_spec._autodetect_num_gpus = _autodetect_num_gpus
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(Path(args.ray_temp_dir).resolve()),
    )

    select_env = "ExploreAgent-v0"
    register_env(
        select_env,
        lambda config: ExploreDrone({
            "env_name": args.env_name,
            "reward_mode": args.reward_mode,
            "max_steps": args.max_steps,
            "gui": False,
        }),
    )
    config = (
        PPOConfig()
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=args.num_workers)
        .training(entropy_coeff=args.entropy_coeff)
        .framework("torch")
    )

    agent = config.build(env=select_env)
    best_reward = float("-inf")
    best_checkpoint = None

    try:
        for n in range(args.iterations):
            result = agent.train()
            mean_reward = result["episode_reward_mean"]

            if n >= args.warmup_iterations and mean_reward > best_reward:
                best_reward = mean_reward
                best_path = checkpoint_dir / "checkpoint_best"
                if best_checkpoint:
                    shutil.rmtree(best_path, ignore_errors=True)
                best_checkpoint = agent.save(str(best_path))
                print(f"Iteration {n + 1}: new best mean reward {mean_reward:.2f}")

            if n % args.save_interval == 0:
                agent.save(str(checkpoint_dir / f"checkpoint_{n:04d}"))

            print(format_training_status(n + 1, result, best_reward, task_limits["max_reward"]))
    finally:
        agent.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
