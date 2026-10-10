# Keenon L1 的 PACE 辨识

Keenon L1 双足腿部执行器参数辨识(当前只做髋 pitch / 髋 roll)。真机数据由 [leg_tool](https://github.com/7huang/leg_tool) 的 pace_chirp 采集、`scripts/to_pace.py` 转换。

## 文件

```
assets/keenon_l1/keenon_l1_pace.urdf                       无网格 URDF(leg_tool 的 scripts/make_pace_urdf.py 生成, 惯性/关节与原 URDF 相同)
source/pace_sim2real/pace_sim2real/tasks/manager_based/keenon_l1/
  __init__.py                                              注册两个任务
  keenon_l1_pace_env_cfg.py                                机器人(URDF 导入)、执行器、边界
scripts/pace/fit_l1.py                                     改写的 fit.py
keenon_l1/training/keenon_l1_pace_training.py              训练用: 执行器配置 + 域随机化(不在 pace_sim2real 包内, 拷到训练工程使用)
```

## 任务

| 任务 | 数据 | 辨识 |
|---|---|---|
| `Isaac-Pace-KeenonL1-HipRoll-v0` | `data/keenon_l1_hip_roll/`(髋 roll 两腿 sym, 头部固定 + 硬腰) | 2 x (armature, 粘滞摩擦, 库伦摩擦, 零偏) + 延迟 |
| `Isaac-Pace-KeenonL1-HipPitchBias-v0` | `data/keenon_l1_hip_pitch/`(髋 pitch 两腿 anti, 头部固定 + 硬腰) | 只辨识 2 个零偏; armature / 粘滞 / 库伦 / 延迟固定为髋 roll 的结果 |

髋 pitch 的动态数据受躯干扭转污染(固定基座单自由度模型误差 30%, 髋 roll 为 7%), 不能辨识 armature 等; 髋 pitch 与髋 roll 是同型号 6416 电机, 借用髋 roll 的结果。零偏主要体现在静态偏移, 受污染影响较小。

## 关键设置

- 仿真 dt = 0.0025 s(400 Hz), decimation 1, 固定 pelvis(髋 roll 采集时腰与手臂峰峰值 < 0.003 rad, 等同固定)。
- 执行器按 `chirp_data_info.json`(to_pace.py 生成)中采集时的 kp/kd 与保持位置生成:
  - `pace` 组: joint_order 中的关节; `hold` 组: 其余 27 个关节, PD 保持在采集位置(两肘约 1.6 rad)。
  - 执行器阻尼 = `KD_SCALE`(0.9) x kd: 多组实测有效阻尼约为指令 kd 的 88-92%, 而 PACE 的粘滞摩擦只能 >= 0。
  - 力矩/速度上限与保持关节的默认 armature 来自电机表(`MOTOR_TABLE`), 手臂电机参数未知, 取 URDF 值与 armature 0.01。
- 髋 roll 边界: armature 0.02-0.20, 粘滞摩擦 0-2, 库伦摩擦 0-2(频域粗估 0.8, 超过 ANYmal 默认上界 0.5), 零偏 ±0.1, 延迟 0-6 步(时间戳实测约 4.7 ms ≈ 2 步)。
- 髋 pitch 任务自动读取 `logs/pace/keenon_l1_hip_roll/` 最新一次运行中编号最大的 `mean_XXX.pt`, 取左右平均作为固定值(延迟取整数步); 也可用环境变量 `PACE_L1_ROLL_RESULT=<mean_XXX.pt 路径>` 指定。找不到时使用频域粗估(armature 0.086, 粘滞 0, 库伦 0.8, 延迟 2 步)并打印提示。

`fit_l1.py` 与 `fit.py` 的区别:
1. 不在 joint_order 中的关节下发采集时的保持位置(原版下发 0), 每一代开始时复位到该位置。
2. `CMAESOptimizer.update_simulator` 会把零偏写到所有执行器组, joint_order 以外的关节会拿到 joint_order[0] 的零偏; 这里在其后把 `hold` 组零偏清零。
3. 检查数据采样间隔与仿真 dt 一致; 结束时在日志目录写 `result.json`(带关节名的物理参数、延迟秒数、kd_scale)。

## 步骤

1. 数据: 2026-10-10 的两组已用 `git add -f` 提交在 `data/keenon_l1_hip_roll/` 与 `data/keenon_l1_hip_pitch/`
   (`data/` 被 .gitignore 忽略, 新增或更新数据时同样需要 `git add -f`)。`*.pt` 由 Git LFS 存储(.gitattributes),
   克隆前须安装 git-lfs, 否则得到的是文本指针文件, `torch.load` 报 `invalid load key 'v'`; 已克隆的仓库执行
   `git lfs install && git lfs pull`。每个目录含 `chirp_data.pt` 与
   `chirp_data_info.json`, 在 leg_tool 目录下从 pace_chirp 运行目录生成:
   ```bash
   python scripts/to_pace.py <运行目录> --dt 0.0025 --joint-order left_hip_roll_joint,right_hip_roll_joint --out <pace-sim2real>/data/keenon_l1_hip_roll/chirp_data.pt
   python scripts/to_pace.py <运行目录> --dt 0.0025 --joint-order left_hip_pitch_joint,right_hip_pitch_joint --out <pace-sim2real>/data/keenon_l1_hip_pitch/chirp_data.pt
   ```
   2026-10-10 的两组数据: leg_tool data 分支 `scripts/data/20261010_093547_m_hip_roll`(髋 roll)与 `20261010_093455_m_hip_pitch`(髋 pitch anti)。
2. 首次运行先看姿态(悬空、两肘弯曲、腿自然下垂):
   ```bash
   python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipRoll-v0 --num_envs 16
   ```
3. 髋 roll 辨识, 查看结果:
   ```bash
   python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipRoll-v0 --num_envs 2048 --headless
   python scripts/pace/plot_trajectory.py --robot_name keenon_l1_hip_roll --plot_trajectory --plot_score
   tensorboard --logdir logs/pace/keenon_l1_hip_roll
   ```
   检查: 分数明显下降并收敛; armature 不贴边界(应在 0.08-0.10 附近); 左右两侧接近; 延迟直方图收拢。若粘滞摩擦贴在下界 0, 说明有效阻尼比 0.9 x kd 还小, 可把 `KD_SCALE` 调到 0.85 重跑。
4. 髋 pitch 零偏(自动使用第 3 步的结果, 开头会打印固定参数的来源):
   ```bash
   python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipPitchBias-v0 --num_envs 512 --headless
   ```
5. 训练: 用 `keenon_l1/training/keenon_l1_pace_training.py` 读两次运行的 `result.json`, 生成髋 pitch/roll 的 `PaceDCMotorCfg`(延迟按秒换算到训练 dt), 并把 `HIP_EVENTS` 加入训练的 EventCfg:
   - 髋 pitch(借用参数): armature 0.06-0.12, 库伦摩擦 0.5-1.2 Nm, 粘滞摩擦 0-0.5, 延迟 2.5-7.5 ms
   - 髋 roll: 按辨识结果设置随机化范围(文件中为占位值)
   - 训练中执行器阻尼同样取 0.9 x kd

## 注意

- 若 URDF 导入失败(无网格连杆), 设 `PACE_L1_URDF` 指向带网格的原 URDF(需 `../meshes/*.STL`)。
- 配置只在无 Isaac Lab 的环境中做过替身测试(读取 info、生成执行器与边界、读取髋 roll 结果), 尚未在 Isaac Lab 中运行。
