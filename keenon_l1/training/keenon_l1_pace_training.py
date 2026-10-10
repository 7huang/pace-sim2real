# 训练(BeyondMimic / Isaac Lab)中使用 PACE 辨识结果: 髋 pitch / 髋 roll 执行器配置 + 域随机化。
#
# 拷到训练工程中, 并确保能 import pace_sim2real.utils(PaceDCMotorCfg); 或把 pace_actuator.py / pace_actuator_cfg.py
# 拷进训练工程(pace-sim2real 文档 Deploying on Hardware 一节)。
#
# 参数来源(fit_l1.py 输出的 result.json):
#   髋 roll : Isaac-Pace-KeenonL1-HipRoll-v0 辨识的 armature / 粘滞摩擦 / 库伦摩擦 / 零偏 / 延迟
#   髋 pitch: armature / 粘滞 / 库伦 / 延迟借用髋 roll(同型号 6416), 零偏由 Isaac-Pace-KeenonL1-HipPitchBias-v0 辨识
# 借用的参数不确定性大, 训练时对髋 pitch 加大随机化范围(armature 0.06-0.12, 库伦摩擦 0.5-1.2 Nm)。
#
# 用法示意:
#   roll = load_pace_result("logs/pace/keenon_l1_hip_roll/<run>/result.json")
#   pitch = load_pace_result("logs/pace/keenon_l1_hip_pitch_bias/<run>/result.json")
#   robot_cfg.actuators["hips"] = hip_actuator_cfg(roll, pitch, physics_dt=env_cfg.sim.dt, max_delay_steps=2)
#   (max_delay_steps 不小于 hip_delay 随机化上界换算的步数; 不随机化延迟时省略)
#   (其它腿部关节从 robot_cfg 的执行器中去掉髋 pitch/roll, 避免同一关节出现在两个执行器组)
#   在 EventCfg 中加入 HIP_EVENTS 的各项(见文件末尾)

from __future__ import annotations

import json

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

from pace_sim2real.utils import PaceDCMotorCfg

# 与 PACE 配置一致: 真机有效阻尼约为指令 kd 的 0.9 倍, 仿真执行器阻尼取 KD_SCALE * kd
KD_SCALE = 0.9

# 训练用 G1 增益(gains_g1_legs.txt): 髋 pitch 40/3, 髋 roll 100/6
HIP_GAINS = {
    "left_hip_pitch_joint": (40.0, 3.0), "right_hip_pitch_joint": (40.0, 3.0),
    "left_hip_roll_joint": (100.0, 6.0), "right_hip_roll_joint": (100.0, 6.0),
}


def load_pace_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    if result.get("kd_scale") not in (None, KD_SCALE):
        raise ValueError(f"{path} 辨识时 kd_scale={result['kd_scale']}, 与训练 KD_SCALE={KD_SCALE} 不一致")
    return result


def hip_actuator_cfg(roll: dict, pitch: dict, physics_dt: float, max_delay_steps: int | None = None) -> PaceDCMotorCfg:
    """
    髋 pitch/roll 的 PaceDCMotorCfg。延迟按秒换算到训练的 physics dt(PACE 为 2.5 ms/步, 训练 dt 通常不同)。
    pitch 结果中 armature/摩擦为固定值(= 髋 roll 均值), 只有零偏是辨识的。
    注意 PaceDCMotor 初始化时延迟取 max_delay: 不做延迟随机化时 max_delay_steps 留空(= 辨识延迟);
    使用 HIP_EVENTS["hip_delay"] 时须传入不小于随机化上界的步数, 由 reset 事件设置每个环境的延迟。
    """
    names = list(HIP_GAINS)
    params = {**roll["joints"], **pitch["joints"]}
    missing = [n for n in names if n not in params]
    if missing:
        raise KeyError(f"结果中缺少 {missing}")
    delay_steps = int(round(roll["delay_s"] / physics_dt))
    return PaceDCMotorCfg(
        joint_names_expr=names,
        saturation_effort=132.0,
        effort_limit=132.0,
        velocity_limit=10.1,
        stiffness={n: HIP_GAINS[n][0] for n in names},
        damping={n: KD_SCALE * HIP_GAINS[n][1] for n in names},
        armature={n: params[n]["armature"] for n in names},
        friction={n: params[n]["friction"] for n in names},
        dynamic_friction={n: params[n]["friction"] for n in names},
        viscous_friction={n: params[n]["viscous_friction"] for n in names},
        encoder_bias={n: params[n]["bias"] for n in names},
        max_delay=delay_steps if max_delay_steps is None else max(max_delay_steps, delay_steps),
    )


##
# 域随机化(事件函数)
##

def randomize_pace_joint_params(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    armature_range: tuple[float, float] | None = None,
    friction_range: tuple[float, float] | None = None,
    viscous_friction_range: tuple[float, float] | None = None,
):
    """
    按绝对值均匀采样 armature / 库伦摩擦(静摩擦 = 动摩擦)/ 粘滞摩擦, 与 PACE 写入仿真的方式相同。
    (Isaac Lab 自带的 randomize_joint_parameters 只改静摩擦, 与 PACE 的库伦摩擦模型不一致。)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(asset.num_joints, device=asset.device)[joint_ids]
    shape = (len(env_ids), len(joint_ids))
    if armature_range is not None:
        value = math_utils.sample_uniform(*armature_range, shape, device=asset.device)
        asset.write_joint_armature_to_sim(value, joint_ids=joint_ids, env_ids=env_ids)
    if friction_range is not None:
        value = math_utils.sample_uniform(*friction_range, shape, device=asset.device)
        # 静摩擦不能小于动摩擦, 先把动摩擦置零再写(与 PACE 的 cma_es.update_simulator 相同)
        asset.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
        asset.write_joint_friction_coefficient_to_sim(value, joint_ids=joint_ids, env_ids=env_ids)
        asset.write_joint_dynamic_friction_coefficient_to_sim(value, joint_ids=joint_ids, env_ids=env_ids)
    if viscous_friction_range is not None:
        value = math_utils.sample_uniform(*viscous_friction_range, shape, device=asset.device)
        asset.write_joint_viscous_friction_coefficient_to_sim(value, joint_ids=joint_ids, env_ids=env_ids)


def randomize_pace_delay(env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, actuator_name: str,
                         delay_range_s: tuple[float, float]):
    """把 PaceDCMotor 执行器组的延迟按秒随机化, 换算成训练 physics dt 下的整数步"""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    dt = env.physics_dt
    lo, hi = int(round(delay_range_s[0] / dt)), int(round(delay_range_s[1] / dt))
    actuator = asset.actuators[actuator_name]
    hi = min(hi, actuator.cfg.max_delay)
    steps = torch.randint(lo, hi + 1, (len(env_ids),), device=asset.device)
    actuator.update_time_lags(steps, env_ids)


# 加到训练的 EventCfg 中(mode="reset": 每个 episode 重新采样)。
# 髋 pitch: 借用髋 roll 的参数, 范围按用户要求放宽; 粘滞摩擦与延迟范围为建议值。
# 髋 roll: 辨识值附近 ±30%(armature)/ ±40%(摩擦), 训练前按 result.json 修改。
HIP_EVENTS = {
    "hip_pitch_params": EventTerm(
        func=randomize_pace_joint_params,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"]),
            "armature_range": (0.06, 0.12),
            "friction_range": (0.5, 1.2),
            "viscous_friction_range": (0.0, 0.5),
        },
    ),
    "hip_roll_params": EventTerm(
        func=randomize_pace_joint_params,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"]),
            "armature_range": (0.06, 0.11),
            "friction_range": (0.5, 1.1),
            "viscous_friction_range": (0.0, 0.3),
        },
    ),
    "hip_delay": EventTerm(
        func=randomize_pace_delay,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": "hips", "delay_range_s": (0.0025, 0.0075)},
    ),
}
