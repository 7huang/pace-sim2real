# PACE 参数辨识(Keenon L1 版), 基于 pace-sim2real 的 scripts/pace/fit.py(© 2025 ETH Zurich, Apache-2.0)。
#
# 与原版的区别:
#   1. 不在 joint_order 中的关节按采集时的保持位置下发目标(原版下发 0), 每一代开始时复位到该位置、速度清零。
#      真机采集用 --hold body, 全身其余关节 PD 保持在采集位置(两肘约 1.6 rad), 仿真须一致。
#   2. CMAESOptimizer.update_simulator 会把零偏写到所有执行器组: 不在 joint_order 中的关节匹配不到, 会拿到
#      joint_order[0] 的零偏。这里在其后把 hold 组的零偏清零(延迟保留, 全部电机共用同一通信链路)。
#   3. 检查数据采样间隔与仿真 dt 一致; 结束时在日志目录写 result.json(带关节名的物理参数, 训练配置直接读取)。
#
# 用法(在 pace-sim2real 根目录):
#   python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipRoll-v0 --num_envs 2048 --headless
#   python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipPitchBias-v0 --num_envs 512 --headless

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PACE 参数辨识(Keenon L1)")
parser.add_argument("--num_envs", type=int, default=2048, help="种群大小 = 并行环境数")
parser.add_argument("--task", type=str, default="Isaac-Pace-KeenonL1-HipRoll-v0", help="任务名")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real import CMAESOptimizer
from pace_sim2real.utils import project_root


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    articulation = env.unwrapped.scene["robot"]
    joint_names = list(articulation.joint_names)
    # 动作向量为全部关节的位置目标, 顺序须与 articulation 相同(JointPositionActionCfg joint_names=[".*"])
    term_joints = list(env.unwrapped.action_manager.get_term("joint_pos")._joint_names)
    if term_joints != joint_names:
        raise RuntimeError(f"动作的关节顺序与 articulation 不一致: {term_joints} vs {joint_names}")

    joint_order = list(env_cfg.sim2real.joint_order)
    sim_joint_ids = torch.tensor([joint_names.index(n) for n in joint_order], device=device)
    hold = dict(env_cfg.sim2real.hold_positions)
    hold_ids = torch.tensor([joint_names.index(n) for n in hold], device=device, dtype=torch.long)
    hold_pos = torch.tensor(list(hold.values()), device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    missing = [n for n in joint_names if n not in hold and n not in joint_order]
    if missing:
        raise RuntimeError(f"这些关节既不在 joint_order 也没有保持位置: {missing}")

    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name
    data = torch.load(data_file)
    time_data = data["time"].to(device)
    target_dof_pos = data["des_dof_pos"].to(device)
    measured_dof_pos = data["dof_pos"].to(device)
    sim_dt = env.unwrapped.sim.cfg.dt
    data_dt = float(time_data[1] - time_data[0])
    if abs(data_dt - sim_dt) > 1e-6:
        raise RuntimeError(f"数据采样间隔 {data_dt} 与仿真 dt {sim_dt} 不一致, 请用 to_pace.py --dt {sim_dt} 重新转换")
    if target_dof_pos.shape[1] != len(joint_order):
        raise RuntimeError(f"数据有 {target_dof_pos.shape[1]} 个关节, joint_order 有 {len(joint_order)} 个")
    time_steps = time_data.shape[0]
    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(num_envs, 1)

    bounds = env_cfg.sim2real.bounds_params.to(device)
    n = len(joint_order)
    names = ["armature", "viscous_friction", "friction", "bias"]
    print(f"[PACE L1] 任务 {args_cli.task}, 数据 {data_file} ({time_steps} 步, dt {sim_dt}), 种群 {num_envs}")
    print(f"[PACE L1] joint_order {joint_order}, 保持关节 {len(hold)} 个")
    for k, name in enumerate(names):
        for j in range(n):
            lo, hi = bounds[k * n + j].tolist()
            print(f"[PACE L1]   {name:17s} {joint_order[j]:24s} [{lo:.4f}, {hi:.4f}]{'  (固定)' if lo == hi else ''}")
    lo, hi = bounds[4 * n].tolist()
    print(f"[PACE L1]   delay(步)                                   [{lo:.1f}, {hi:.1f}]{'  (固定)' if lo == hi else ''}")

    opt = CMAESOptimizer(
        bounds=bounds,
        population_size=num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=device,
        epsilon=env_cfg.sim2real.cmaes.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
    )

    # 每一代: 复位、写入候选参数, 再把保持关节放回采集位置并清掉它们被误写的零偏
    base_actions = torch.zeros(env.action_space.shape, device=device)
    if len(hold):
        base_actions[:, hold_ids] = hold_pos

    def reset_robot():
        env.reset()
        opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)
        if len(hold):
            articulation.write_joint_position_to_sim(hold_pos, joint_ids=hold_ids)
            articulation.write_joint_velocity_to_sim(torch.zeros_like(hold_pos), joint_ids=hold_ids)
        if "hold" in articulation.actuators:
            act = articulation.actuators["hold"]
            act.update_encoder_bias(torch.zeros((num_envs, act.num_joints), device=device))

    counter = 0
    with torch.inference_mode():
        reset_robot()
    while simulation_app.is_running():
        with torch.inference_mode():
            opt.tell(articulation.data.joint_pos[:, sim_joint_ids], measured_dof_pos[counter, :].unsqueeze(0).repeat(num_envs, 1))
            actions = base_actions.clone()
            actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(num_envs, 1)
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)")
            if counter >= time_steps:
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                reset_robot()

    # 带关节名的结果, 供训练配置读取(参数为 CMA-ES 均值, 与 mean_XXX.pt 相同)
    # 不用 opt.get_best_sim_params(): 它把 CPU 上的均值与 GPU 上的 bounds 相加, 在 cuda 上会报设备不一致
    mean = torch.tensor(opt.optimizer._mean, device=device, dtype=bounds.dtype)
    best = opt._params_to_sim_params(mean).cpu()
    delay_steps = int(best[4 * n])
    result = {
        "task": args_cli.task,
        "data": str(data_file),
        "sim_dt": sim_dt,
        "kd_scale": getattr(env_cfg.sim2real, "kd_scale", None),
        "joints": {joint_order[j]: {name: float(best[k * n + j]) for k, name in enumerate(names)} for j in range(n)},
        "delay_steps": delay_steps,
        "delay_s": delay_steps * sim_dt,
        "fixed": {f"{names[k]}/{joint_order[j]}": bool(bounds[k * n + j, 0] == bounds[k * n + j, 1])
                  for k in range(4) for j in range(n)},
    }
    out = opt.writer.log_dir + "/result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[PACE L1] 结果已写入 {out}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    opt.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
