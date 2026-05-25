import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import ray
import torch
from ray._private import resource_spec
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.tune.registry import register_env

from explore_agent.envs.exploring_gym import ExploreDrone
from explore_agent.envs.reward_config import CHECKPOINT_REWARD


def _autodetect_num_gpus():
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPG on the 2D checkpoint exploration task.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=0)
    parser.add_argument("--env-name", default="2d_checkpoint_exploration")
    parser.add_argument("--reward-mode", default="coverage")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--checkpoint-dir", default="tmp/ddpg_2d_checkpoint_exploration")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--ray-temp-dir", default=str(Path(tempfile.gettempdir()) / "aiar_ray"))
    parser.add_argument(
        "--layout-path",
        "--rooms-layout-path",
        dest="rooms_layout_path",
        default="",
        help="Optional .npz layout exported by the local layout editor.",
    )
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--spawn-mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.002)
    parser.add_argument("--exploration-initial-scale", type=float, default=1.0)
    parser.add_argument("--exploration-final-scale", type=float, default=0.02)
    parser.add_argument("--exploration-scale-timesteps", type=int, default=10000)
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint):
    checkpoint = Path(checkpoint)
    if (checkpoint / "rllib_checkpoint.json").exists():
        return checkpoint

    candidates = sorted(
        checkpoint.glob("checkpoint_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "rllib_checkpoint.json").exists():
            return candidate

    return checkpoint


def make_env_config(args):
    return {
        "env_name": args.env_name,
        "reward_mode": args.reward_mode,
        "max_steps": args.max_steps,
        "gui": False,
        "render_mode": None,
        "rooms_layout_path": args.rooms_layout_path,
        "spawn_mode": args.spawn_mode,
        "spawn_index": args.spawn_index,
    }


def get_task_limits(args):
    env = ExploreDrone(make_env_config(args))
    try:
        return {
            "n_checkpoints": env.env.n_goals,
            "max_reward": env.env.n_goals * CHECKPOINT_REWARD,
            "observation_shape": env.observation_space.shape,
            "checkpoint_layout_source": env.env.rooms_layout_source_path or "built-in",
        }
    finally:
        env.close()


def print_training_context(args, task_limits):
    print("\nTraining DDPG")
    print(f"  environment: {args.env_name}")
    print(f"  reward mode: {args.reward_mode}")
    print(f"  max steps per episode: {args.max_steps}")
    print(f"  rollout workers: {args.num_workers}")
    print(f"  requested GPUs: {args.num_gpus}")
    print(f"  observation shape: {task_limits['observation_shape']}")
    print(f"  checkpoint layout: {task_limits['checkpoint_layout_source']}")
    print(f"  spawn mode: {args.spawn_mode}")
    print(
        "  DDPG update: "
        f"train batch {args.train_batch_size}, actor lr {args.actor_lr:g}, "
        f"critic lr {args.critic_lr:g}, gamma {args.gamma:g}, tau {args.tau:g}"
    )
    print(
        "  DDPG exploration: "
        f"OU noise scale {args.exploration_initial_scale:g} -> "
        f"{args.exploration_final_scale:g} over {args.exploration_scale_timesteps} steps"
    )
    print(
        "  coverage task: "
        f"{task_limits['n_checkpoints']} checkpoints x {CHECKPOINT_REWARD:.2f} reward = "
        f"{task_limits['max_reward']:.2f} max reward"
    )
    print("\nLog columns")
    print("  iter: completed DDPG training iteration")
    print("  reward min/mean/max: episode return statistics from the latest training result")
    print("  score%: mean reward as a percentage of the checkpoint reward maximum")
    print("  len: mean episode length in environment steps")
    print("  best: best mean reward saved after warmup iterations\n")
    sys.stdout.flush()


def format_training_status(iteration, result, best_reward, max_reward):
    min_reward = result["episode_reward_min"]
    mean_reward = result["episode_reward_mean"]
    max_seen_reward = result["episode_reward_max"]
    mean_len = result["episode_len_mean"]
    best = best_reward if best_reward != float("-inf") else 0.0
    score_percent = 100.0 * mean_reward / max_reward if max_reward > 0 else 0.0
    return (
        f"iter {iteration:4d} | "
        f"reward {min_reward:7.2f}/{mean_reward:7.2f}/{max_seen_reward:7.2f} | "
        f"score {score_percent:6.1f}% | "
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
    register_env(select_env, lambda config: ExploreDrone(make_env_config(args)))

    exploration_config = DDPGConfig().exploration_config.copy()
    exploration_config.update({
        "initial_scale": args.exploration_initial_scale,
        "final_scale": args.exploration_final_scale,
        "scale_timesteps": args.exploration_scale_timesteps,
    })

    config = (
        DDPGConfig()
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=args.num_workers)
        .training(
            train_batch_size=args.train_batch_size,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            tau=args.tau,
        )
        .exploration(exploration_config=exploration_config)
        .framework("torch")
    )

    agent = config.build(env=select_env)
    if args.resume_from:
        resume_checkpoint = resolve_checkpoint_path(args.resume_from)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        agent.restore(str(resume_checkpoint))
        print(f"Restored training state from {resume_checkpoint}")

    best_reward = float("-inf")

    try:
        for n in range(args.iterations):
            result = agent.train()
            mean_reward = result["episode_reward_mean"]

            if n >= args.warmup_iterations and mean_reward > best_reward:
                best_reward = mean_reward
                best_path = checkpoint_dir / "checkpoint_best"
                shutil.rmtree(best_path, ignore_errors=True)
                agent.save(str(best_path))
                print(f"Iteration {n + 1}: new best mean reward {mean_reward:.2f}")

            if n % args.save_interval == 0:
                agent.save(str(checkpoint_dir / f"checkpoint_{n:04d}"))

            print(format_training_status(n + 1, result, best_reward, task_limits["max_reward"]))
    finally:
        agent.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
