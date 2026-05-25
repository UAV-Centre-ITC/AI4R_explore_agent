# Windows Setup Notes

These notes are for students using Windows with Anaconda Prompt or a Conda-enabled PowerShell. The main commands in the repository were tested on Ubuntu/Linux, but the Conda workflow should be similar on Windows.

## 1. Install Conda

Install Miniconda or Anaconda using the official Windows guide:

<https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>

Close and reopen Anaconda Prompt after installation so that `conda` is available.

## 2. Clone the Repository

Choose a working folder first. This example uses `Documents`:

```bat
cd %USERPROFILE%\Documents
mkdir AI4R
cd AI4R
git clone https://github.com/UAV-Centre-ITC/AI4R_explore_agent.git
cd AI4R_explore_agent
cd Exploring_agent_DRL
```

If `git` is not available, install Git for Windows and open a new Anaconda Prompt.

Windows paths often use backslashes, for example:

```bat
C:\Users\YourName\Documents\AI4R\AI4R_explore_agent\Exploring_agent_DRL
```

If a path contains spaces, wrap it in quotes:

```bat
cd "C:\Users\YourName\My Documents\AI4R"
```

## 3. Create the Conda Environment

CPU environment:

```bat
conda env create -f environment.yml
conda activate aiar-rl-explore
```

NVIDIA GPU environment:

```bat
conda env create -f environment-gpu.yml
conda activate aiar-rl-explore-gpu
```

Use the GPU environment only if the NVIDIA driver is installed and `nvidia-smi` works.

## 4. Quick Check

Run this command from the `Exploring_agent_DRL` folder:

```bat
python run_assignment.py check --steps 1000
```

If this completes, continue with the training and rollout commands in the main README.
