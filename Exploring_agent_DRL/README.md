## AIAR RL practical assignment: 2D exploring robot

This folder contains the runnable code for the practical assignment. The maintained path is PPO with Ray RLlib, Gymnasium, PyTorch, and Pygame.

## Windows-first setup with conda

These instructions are written for a Windows machine using Anaconda Prompt or Miniconda Prompt.

First open **Anaconda Prompt**, then change to this folder. For example:

```bat
cd C:\path\to\AIAR_RL_practical_assignment_2_2025\Exploring_agent_DRL
```

If the repository path contains spaces, wrap it in quotes:

```bat
cd "C:\path with spaces\AIAR_RL_practical_assignment_2_2025\Exploring_agent_DRL"
```

## Environment choice

Use one of the following environments:

- CPU-only: `environment.yml`, environment name `aiar-rl-explore`
- NVIDIA GPU: `environment-gpu.yml`, environment name `aiar-rl-explore-gpu`

The CPU environment is the safest default for teaching and reproducibility. The GPU environment is useful for faster neural-network updates on machines with an NVIDIA GPU and working drivers.

## CPU-only conda setup

Use this if the supervisor does not need GPU acceleration, or if CUDA/PyTorch GPU setup is not already working.

```bat
set PYTHONNOUSERSITE=1
conda env create -f environment.yml
conda activate aiar-rl-explore
```

If you already created the environment and only changed dependencies:

```bat
set PYTHONNOUSERSITE=1
conda env update -f environment.yml --prune
conda activate aiar-rl-explore
```

## GPU conda setup

Use this only if the Windows machine has an NVIDIA GPU and `nvidia-smi` works in Anaconda Prompt:

```bat
nvidia-smi
set PYTHONNOUSERSITE=1
conda env create -f environment-gpu.yml
conda activate aiar-rl-explore-gpu
```

Verify CUDA before training:

```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Only use `--num-gpus 1` after `torch.cuda.is_available()` prints `True`.

## Quick environment check

Run this after activating either `aiar-rl-explore` or `aiar-rl-explore-gpu`:

```bat
python -c "from explore_agent.envs.exploring_gym import ExploreDrone; env = ExploreDrone({'gui': False, 'env_name': 'playground', 'reward_mode': 'continuous', 'max_steps': 5}); obs, _ = env.reset(); print('observation shape:', obs.shape); obs, reward, terminated, truncated, info = env.step([0.0, 0.0]); print(round(float(reward), 4), terminated, truncated, info)"
```

## Manual driving

Use this first to confirm Pygame rendering and the map work:

```bat
python Explore-Agent\explore_agent\envs\start_human.py
```

Use the arrow keys to drive. Press `r` to reset.

## Train PPO

CPU-only short run:

```bat
python Explore_PPO_agent_training.py --iterations 20 --num-workers 1 --num-gpus 0
```

CPU-only longer run:

```bat
python Explore_PPO_agent_training.py --iterations 500 --num-workers 2 --num-gpus 0
```

GPU training run, only from the `aiar-rl-explore-gpu` environment:

```bat
python Explore_PPO_agent_training.py --iterations 500 --num-workers 2 --num-gpus 1
```

Checkpoints are written to `tmp/ppo/`. The best checkpoint is saved at:

```text
tmp/ppo/checkpoint_best
```

## Test a trained PPO checkpoint

```bat
python explore_agent_rollout.py --checkpoint tmp/ppo/checkpoint_best
```

The rollout opens the Pygame window and prints the cumulative reward when the episode ends.

Ray uses a short temporary directory under the system temp folder by default. This avoids path-length issues when the repository is stored in a deeply nested directory.

## Visualize training metrics

Ray writes training logs under the user home `ray_results` folder by default. On Windows:

```bat
python -m tensorboard.main --logdir %USERPROFILE%\ray_results
```

Open the TensorBoard URL printed in the terminal.
