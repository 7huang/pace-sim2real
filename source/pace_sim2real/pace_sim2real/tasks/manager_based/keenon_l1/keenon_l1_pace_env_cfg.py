# Keenon L1 的 PACE 配置(参照 pace_sim2real 的 anymal_pace_env_cfg.py)
#
# 两个任务:
#   Isaac-Pace-KeenonL1-HipRoll-v0       用髋 roll 数据辨识髋 roll 的 armature / 粘滞摩擦 / 库伦摩擦 / 零偏 / 延迟
#   Isaac-Pace-KeenonL1-HipPitchBias-v0  髋 pitch 的 armature / 粘滞 / 库伦 / 延迟固定为髋 roll 的辨识结果(同型号 6416),
#                                        只用髋 pitch 数据辨识零偏(吊装/头部固定时躯干仍会动, 髋 pitch 的动态数据不可信)
#
# 数据: <pace 根目录>/data/<robot_name>/chirp_data.pt 与同目录的 chirp_data_info.json, 由 leg_tool 的 scripts/to_pace.py
# 生成(--dt 0.0025)。info 中记录了采集时每个关节的 kp/kd 与保持位置, 本配置据此生成执行器, 保证与真机一致。

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.utils import PaceDCMotorCfg, project_root

##
# 机器人
##

# 无网格 URDF(leg_tool 的 scripts/make_pace_urdf.py 生成), 可用环境变量 PACE_L1_URDF 指向其它 URDF
URDF_PATH = Path(os.environ.get("PACE_L1_URDF", str(project_root() / "assets" / "keenon_l1" / "keenon_l1_pace.urdf")))

# 实测有效阻尼约为指令 kd 的 88-92%(多组数据, 腿部电机)。PACE 的粘滞摩擦只能 >= 0, 无法表达"比 kd 小",
# 因此仿真中执行器阻尼取 KD_SCALE * kd, 粘滞摩擦在此基础上辨识。训练时须使用同样的 KD_SCALE。
KD_SCALE = 0.9

# 各关节电机参数(用户电机表; 并联踝/腰的力矩限幅取 C 端关节空间限幅; 手臂取 URDF):
#   (关节名正则, 力矩上限 Nm, 最大角速度 rad/s, 默认 armature kg*m^2, 默认库伦摩擦 Nm)
# armature / 摩擦只是保持关节的默认值, 被辨识的关节由 PACE 覆盖。
MOTOR_TABLE = [
    (r".*_hip_(pitch|roll)_joint", 132.0, 10.1, 0.095, 0.8),   # 6416
    (r".*_knee_joint", 132.0, 10.1, 0.095, 0.5),               # 6416
    (r".*_hip_yaw_joint", 70.0, 12.8, 0.058, 0.3),             # 6408-30H
    (r".*_ankle_pitch_joint", 60.0, 3.9, 0.05, 0.5),           # 4310 并联
    (r".*_ankle_roll_joint", 20.0, 3.9, 0.05, 0.5),            # 4310 并联
    (r"waist_yaw_joint", 45.0, 11.0, 0.016, 0.3),              # 6408-16H
    (r"waist_(roll|pitch)_joint", 60.0, 5.7, 0.066, 0.3),      # 4315 并联
    (r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist_roll)_joint", 36.0, 7.85, 0.01, 0.1),
    (r".*_wrist_(pitch|yaw)_joint", 12.0, 23.0, 0.01, 0.1),
]

# 髋 roll 的频域粗估(2026-10-10 数据, 延迟取 2 步): 在没有 PACE 辨识结果时作为髋 pitch 的固定值
HIP_ROLL_COARSE = {"armature": 0.086, "viscous_friction": 0.0, "friction": 0.8, "delay": 2.0}

KEENON_L1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(URDF_PATH),
        fix_base=False,  # 由 PaceSim2realEnvCfg 设置 articulation_props.fix_root_link = True 固定 pelvis
        merge_fixed_joints=False,
        make_instanceable=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,  # PhysX 默认 0.05, 会引入额外阻尼
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # 悬空: 腿长约 0.85 m, 1.2 m 高度下脚不触地
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.2), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    actuators={},  # 由 configure_robot() 按采集数据(chirp_data_info.json)生成
)


##
# 辅助函数(不依赖仿真, 可单独测试)
##

def motor_params(name: str) -> tuple[float, float, float, float]:
    """关节名 -> (力矩上限, 最大角速度, 默认 armature, 默认库伦摩擦)"""
    for pattern, effort, vel, armature, friction in MOTOR_TABLE:
        if re.fullmatch(pattern, name):
            return effort, vel, armature, friction
    raise KeyError(f"MOTOR_TABLE 中没有关节 {name}")


def load_info(data_file: Path) -> dict:
    """to_pace.py 生成的 <data>_info.json"""
    info_path = data_file.with_name(data_file.stem + "_info.json")
    if not info_path.exists():
        raise FileNotFoundError(f"找不到 {info_path}(to_pace.py 生成, 须与 {data_file.name} 放在一起)")
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def joint_setup(info: dict, joint_order: list[str]) -> tuple[dict, dict]:
    """返回 ({关节名: (kp, kd)}, {关节名: 起始/保持位置}), 包含 joint_order 与真机上被 PD 保持的全部关节"""
    if list(info["joint_order"]) != list(joint_order):
        raise ValueError(f"数据的 joint_order {info['joint_order']} 与配置 {joint_order} 不一致")
    if info.get("limp_joints"):
        raise ValueError(f"采集时有无力的并联关节 {info['limp_joints']}, 本配置要求全部关节受控(--hold body)")
    gains = {n: (float(g["kp"]), float(g["kd"])) for n, g in info["gains"].items()}
    pos = {n: float(p) for n, p in info["initial_pos"].items()}
    for n, h in info["held_not_in_order"].items():
        gains[n] = (float(h["kp"]), float(h["kd"]))
        pos[n] = float(h["pos"])
    return gains, pos


def latest_pace_result(robot_name: str) -> Path | None:
    """logs/pace/<robot_name>/ 下最新一次运行中编号最大的 mean_XXX.pt(与 plot_trajectory.py 的选择方式相同)"""
    log_dir = project_root() / "logs" / "pace" / robot_name
    if not log_dir.exists():
        return None
    runs = [p for p in log_dir.iterdir() if p.is_dir() and any(p.glob("mean_*.pt"))]
    if not runs:
        return None
    run = max(runs, key=lambda p: p.stat().st_mtime)
    return max(run.glob("mean_*.pt"), key=lambda p: int(re.match(r"mean_(\d+)\.pt", p.name).group(1)))


def hip_roll_result() -> tuple[dict, str]:
    """
    髋 roll 的 PACE 辨识结果(左右平均), 用作髋 pitch 的固定值。
    mean_XXX.pt 为物理量参数向量 [armature(2) | 粘滞摩擦(2) | 库伦摩擦(2) | 零偏(2) | 延迟(1)]。
    环境变量 PACE_L1_ROLL_RESULT 可指定文件; 找不到结果时使用 HIP_ROLL_COARSE 并给出提示。
    """
    path = os.environ.get("PACE_L1_ROLL_RESULT")
    path = Path(path) if path else latest_pace_result("keenon_l1_hip_roll")
    if path is None:
        return dict(HIP_ROLL_COARSE), "频域粗估 HIP_ROLL_COARSE(未找到髋 roll 的 PACE 结果)"
    p = torch.load(path, map_location="cpu").flatten().double()
    if p.numel() != 9:
        raise ValueError(f"{path} 有 {p.numel()} 个参数, 髋 roll 结果应为 9 个(2 关节 x 4 + 延迟)")
    result = {
        "armature": float(p[0:2].mean()),
        "viscous_friction": float(p[2:4].mean()),
        "friction": float(p[4:6].mean()),
        "delay": float(int(p[8])),  # PACE 内部按 int 截断, 固定值也取整数步
    }
    return result, str(path)


def configure_robot(env_cfg: PaceSim2realEnvCfg, fixed: dict | None = None) -> None:
    """
    按采集数据生成执行器与初始姿态:
      pace: joint_order 中的关节(单独一组, 避免 PACE 把零偏写到其它关节上, 见 fit_l1.py)
      hold: 其余关节, 按真机增益保持在采集时的位置
    fixed: 髋 pitch 任务中写入的固定参数(也作为 pace 组的初值); 此时被保持的髋 roll 也使用这组辨识值
    """
    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    info = load_info(data_file)
    if abs(float(info["dt"]) - env_cfg.sim.dt) > 1e-9:
        raise ValueError(f"数据 dt={info['dt']} 与仿真 dt={env_cfg.sim.dt} 不一致, 请用 to_pace.py --dt {env_cfg.sim.dt} 重新转换")
    joint_order = list(env_cfg.sim2real.joint_order)
    gains, pos = joint_setup(info, joint_order)

    def group(names: list[str], use_fixed: bool) -> PaceDCMotorCfg:
        effort, vel, arm, fric, visc = {}, {}, {}, {}, {}
        for n in names:
            e, v, a, c = motor_params(n)
            effort[n], vel[n] = e, v
            arm[n], fric[n], visc[n] = a, c, 0.0
            if fixed is not None and (use_fixed or re.fullmatch(r".*_hip_roll_joint", n)):
                arm[n], fric[n], visc[n] = fixed["armature"], fixed["friction"], fixed["viscous_friction"]
        return PaceDCMotorCfg(
            joint_names_expr=names,
            saturation_effort=max(effort.values()),
            effort_limit=effort,
            velocity_limit=vel,
            stiffness={n: gains[n][0] for n in names},
            damping={n: KD_SCALE * gains[n][1] for n in names},
            armature=arm,
            friction=fric,           # 静摩擦 = 动摩擦: 库伦摩擦
            dynamic_friction=fric,
            viscous_friction=visc,
            encoder_bias={n: 0.0 for n in names},
            max_delay=10,
        )

    held = [n for n in gains if n not in joint_order]
    actuators = {"pace": group(joint_order, True)}
    if held:
        actuators["hold"] = group(held, False)
    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        actuators=actuators,
        init_state=env_cfg.scene.robot.init_state.replace(joint_pos=dict(pos)),
    )
    # fit_l1.py 用来把保持关节下发到采集时的位置; kd_scale 写入 result.json
    env_cfg.sim2real.hold_positions = {n: pos[n] for n in held}
    env_cfg.sim2real.kd_scale = KD_SCALE


##
# PACE 配置
##

@configclass
class KeenonL1HipRollPaceCfg(PaceCfg):
    """髋 roll: 两腿镜像(sym)数据, 辨识 2 x (armature, 粘滞摩擦, 库伦摩擦, 零偏) + 全局延迟 = 9 个参数"""
    robot_name: str = "keenon_l1_hip_roll"
    data_dir: str = "keenon_l1_hip_roll/chirp_data.pt"
    joint_order: list[str] = ["left_hip_roll_joint", "right_hip_roll_joint"]
    bounds_params: torch.Tensor = torch.zeros((9, 2))
    hold_positions: dict = {}  # configure_robot() 填写
    kd_scale: float = 1.0      # configure_robot() 填写

    def __post_init__(self):
        n = len(self.joint_order)
        b = self.bounds_params
        b[0:n, 0], b[0:n, 1] = 0.02, 0.20                  # armature [kg m^2], 电机表 0.095, 频域粗估 0.08-0.09
        b[n:2 * n, 0], b[n:2 * n, 1] = 0.0, 2.0            # 粘滞摩擦 [Nm s/rad](在 KD_SCALE*kd 之外)
        b[2 * n:3 * n, 0], b[2 * n:3 * n, 1] = 0.0, 2.0    # 库伦摩擦 [Nm], 粗估 0.8, 超出 ANYmal 默认上界 0.5
        b[3 * n:4 * n, 0], b[3 * n:4 * n, 1] = -0.1, 0.1   # 零偏 [rad]
        b[4 * n, 0], b[4 * n, 1] = 0.0, 6.0                # 延迟 [步, 2.5 ms/步], 时间戳实测约 4.7 ms


@configclass
class KeenonL1HipPitchBiasPaceCfg(PaceCfg):
    """髋 pitch: 只辨识零偏, 其余参数固定为髋 roll 的结果(上下界相同即为固定, 见 KeenonL1HipPitchBiasPaceEnvCfg)"""
    robot_name: str = "keenon_l1_hip_pitch_bias"
    data_dir: str = "keenon_l1_hip_pitch/chirp_data.pt"
    joint_order: list[str] = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
    bounds_params: torch.Tensor = torch.zeros((9, 2))
    hold_positions: dict = {}  # configure_robot() 填写
    kd_scale: float = 1.0      # configure_robot() 填写

    def __post_init__(self):
        n = len(self.joint_order)
        self.bounds_params[3 * n:4 * n, 0], self.bounds_params[3 * n:4 * n, 1] = -0.1, 0.1


##
# 场景与环境
##

@configclass
class KeenonL1PaceSceneCfg(PaceSim2realSceneCfg):
    robot: ArticulationCfg = KEENON_L1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class KeenonL1HipRollPaceEnvCfg(PaceSim2realEnvCfg):
    scene: KeenonL1PaceSceneCfg = KeenonL1PaceSceneCfg()
    sim2real: PaceCfg = KeenonL1HipRollPaceCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.0025  # 400 Hz, 与 to_pace.py --dt 一致
        self.decimation = 1
        configure_robot(self)


@configclass
class KeenonL1HipPitchBiasPaceEnvCfg(PaceSim2realEnvCfg):
    scene: KeenonL1PaceSceneCfg = KeenonL1PaceSceneCfg()
    sim2real: PaceCfg = KeenonL1HipPitchBiasPaceCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.0025
        self.decimation = 1
        fixed, source = hip_roll_result()
        print(f"[PACE L1] 髋 pitch 固定参数来自 {source}: {fixed}")
        n = len(self.sim2real.joint_order)
        b = self.sim2real.bounds_params
        b[0:n, :] = fixed["armature"]
        b[n:2 * n, :] = fixed["viscous_friction"]
        b[2 * n:3 * n, :] = fixed["friction"]
        b[4 * n, :] = fixed["delay"]
        configure_robot(self, fixed)
