import argparse
import shutil
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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_gpus > 0 and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training was requested with --num-gpus > 0, but this environment has CPU-only PyTorch. "
            "Install a CUDA-enabled PyTorch build or run with --num-gpus 0."
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        .framework("torch")
    )

    agent = config.build(env=select_env)
    status = "{:4d} reward {:8.2f}/{:8.2f}/{:8.2f} len {:6.2f}"
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

            print(status.format(
                n + 1,
                result["episode_reward_min"],
                mean_reward,
                result["episode_reward_max"],
                result["episode_len_mean"],
            ))
    finally:
        agent.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
