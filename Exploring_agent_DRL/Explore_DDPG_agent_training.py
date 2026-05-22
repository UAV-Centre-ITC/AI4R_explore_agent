import os
import shutil
import ray

from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig



resource_spec._autodetect_num_gpus = _autodetect_num_gpus
ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=r"..\tmp\ray")


chkpt_root = "tmp/ppo/checkpoint_{}"
ray_results = "{}/ray_results/".format(os.getenv("HOME"))


select_env = "ExploreAgent-v0"
register_env(select_env, lambda config: ExploreDrone({"env_name": "playground",  "reward_mode": "continuous"}))
config = DDPGConfig().resources(num_gpus=0).rollouts(num_rollout_workers=24)

agent = config.build(env=select_env)
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}"

n_iter = 4000

for n in range(n_iter):
    result = agent.train()
    if n%50 ==0:
        chkpt_file = agent.save(chkpt_root)
    print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
            ))
