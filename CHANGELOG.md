# Changelog

## 2026-05-22 - Source release preparation

Prepared the assignment repository so it can be cloned and run on another computer without carrying local checkpoints, build artifacts, or machine-specific setup.

### Setup and environment changes

- Added `Exploring_agent_DRL/environment.yml` for CPU-only Conda setup.
- Added `Exploring_agent_DRL/environment-gpu.yml` for NVIDIA GPU Conda setup.
- Switched the documented setup path to Conda/Anaconda Prompt instead of micromamba.
- Added separate CPU and GPU environment names:
  - CPU: `aiar-rl-explore`
  - GPU: `aiar-rl-explore-gpu`
- Pinned compatibility-sensitive packages, including `setuptools<81`, `mkl<2024`, PyTorch 2.0, and Pydantic 1.x.
- Added GPU setup dependencies needed by the tested CUDA environment.

### README and usage documentation

- Rewrote `Exploring_agent_DRL/README.md` as a Windows-first guide for a supervisor or student starting from a fresh clone.
- Documented CPU-only setup, GPU setup, CUDA verification, environment smoke test, manual driving, PPO training, PPO rollout, and TensorBoard visualization.
- Added explicit training commands for short CPU tests, longer CPU training, and GPU training.
- Added a root `README.md` that points users to the exercise guide and assignment handout.

### PPO training changes

- Updated `Exploring_agent_DRL/Explore_PPO_agent_training.py` to accept command-line arguments instead of requiring source edits for common settings.
- Added options for iterations, save interval, warmup iterations, rollout workers, GPU count, environment name, reward mode, checkpoint directory, and Ray temporary directory.
- Added an explicit CUDA availability check when `--num-gpus` is greater than zero.
- Kept CPU-only training as the default with `--num-gpus 0`.
- Added best-checkpoint saving under `tmp/ppo/checkpoint_best`.
- Moved Ray temporary files to a short system temp path by default to reduce Windows path-length issues.

### Rollout changes

- Updated `Exploring_agent_DRL/explore_agent_rollout.py` to accept command-line arguments.
- Added a default checkpoint path of `tmp/ppo/checkpoint_best`.
- Added checkpoint path resolution for RLlib checkpoint directories.
- Added configurable rollout steps, sleep time, environment name, reward mode, and Ray temporary directory.

### Explore-Agent package changes

- Updated `Exploring_agent_DRL/Explore-Agent/setup.py` dependencies so the editable install includes required runtime packages.
- Added `pillow` for image loading support.
- Pinned `pydantic==1.10.15` for compatibility with Ray/RLlib 2.5.1.

### Gymnasium environment fixes

- Updated `Exploring_agent_DRL/Explore-Agent/explore_agent/envs/exploring_gym.py` so observations are clipped to the declared observation space.
- Returned observations as `np.float32`, matching the declared Gymnasium space.
- Clipped actions to the declared action range before applying them to the drone.
- These changes make the environment behavior match the declared `Box` spaces expected by RLlib.

### Repository hygiene

- Initialized the local git repository on branch `main`.
- Added the GitHub remote `https://github.com/UAV-Centre-ITC/AI4R_explore_agent.git`.
- Added `.gitignore` to keep local outputs out of version control:
  - RLlib/Ray checkpoints and results
  - `Exploring_agent_DRL/tmp/`
  - TensorBoard logs
  - Python cache files
  - package build artifacts
  - local virtual/Conda environments
  - exported experiment arrays and pickle files
- Added `.gitattributes` for consistent text and binary file handling.
- Pushed the source-only release to GitHub.

### Verification performed

- Created and tested the GPU Conda environment locally.
- Verified PyTorch CUDA availability on the local NVIDIA GPU machine.
- Ran a short PPO GPU smoke training run.
- Verified `pip check` passed after dependency fixes.
- Confirmed generated folders such as `tmp/`, `__pycache__/`, and `explore_agent.egg-info/` were ignored by git.
- Confirmed no checkpoint/cache paths were tracked before pushing.

### Remaining notes

- Local checkpoints and generated outputs still exist on the workstation if they were created during testing, but they are ignored by git and were not pushed.
- The maintained training path is PPO. The DDPG script remains in the repository as source code but was not the focus of the reproducibility update.
