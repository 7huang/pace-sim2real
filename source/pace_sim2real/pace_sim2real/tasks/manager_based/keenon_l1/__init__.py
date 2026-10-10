# Keenon L1 的 PACE 任务(放在 pace_sim2real/tasks/manager_based/keenon_l1/ 下, 由 tasks/__init__.py 的 import_packages 自动注册)

import gymnasium as gym

gym.register(
    id="Isaac-Pace-KeenonL1-HipRoll-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.keenon_l1_pace_env_cfg:KeenonL1HipRollPaceEnvCfg"},
)

gym.register(
    id="Isaac-Pace-KeenonL1-HipPitchBias-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.keenon_l1_pace_env_cfg:KeenonL1HipPitchBiasPaceEnvCfg"},
)
