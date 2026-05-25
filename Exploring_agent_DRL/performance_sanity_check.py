import argparse
import os
import time

from explore_agent.envs.exploring_gym import ExploreDrone


def make_env(render_mode, render_fps=30):
    return ExploreDrone({
        "env_name": "2d_checkpoint_exploration",
        "reward_mode": "coverage",
        "max_steps": 400,
        "gui": render_mode == "human",
        "render_mode": render_mode,
        "render_fps": render_fps,
    })


def run_headless_check(steps):
    env = make_env(render_mode=None)
    env.reset()

    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - t0

    print("Headless stepping")
    print(f"  steps: {steps}")
    print(f"  steps per second: {steps / elapsed:.1f}")
    print(f"  average step time: {1000.0 * elapsed / steps:.3f} ms")
    print(f"  pygame display initialized: {env.display_initialized}")
    print(f"  rendering active: {env.render_mode is not None}")
    env.close()


def run_visual_check(steps, render_fps):
    env = make_env(render_mode="human", render_fps=render_fps)
    env.reset()

    t0 = time.perf_counter()
    try:
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            env.render()
            if terminated or truncated:
                env.reset()
    finally:
        elapsed = time.perf_counter() - t0
        print("\nVisual rendering")
        print(f"  requested FPS cap: {render_fps}")
        print(f"  rendered frames: {steps}")
        print(f"  measured FPS: {steps / elapsed:.1f}")
        print(f"  pygame display initialized: {env.display_initialized}")
        print(f"  rendering active: {env.render_mode == 'human'}")
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check environment stepping and Pygame render throttling.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--visual-steps", type=int, default=120)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument(
        "--dummy-video",
        action="store_true",
        help="Set SDL_VIDEODRIVER=dummy before the visual check for headless machines.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_headless_check(args.steps)

    if args.skip_visual:
        return
    if args.dummy_video:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    run_visual_check(args.visual_steps, args.render_fps)


if __name__ == "__main__":
    main()
