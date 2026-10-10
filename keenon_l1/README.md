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

## 使用说明

流程: 髋 roll 全参数辨识 → 固定这些参数、只辨识髋 pitch 零偏 → 把结果用到策略训练。

### 0. 环境要求

- Ubuntu 22.04, NVIDIA GPU(显存 ≥ 8 GB, 建议 16 GB)。
- **Isaac Sim 5.0 + 对应的 Isaac Lab**(PACE 官方测试: Isaac Lab 0.46.2, Python 3.11)。不要用 4.5: 4.5 不支持关节粘滞摩擦, 只打印警告后静默忽略, 而本配置依赖它(阻尼 = 0.9 x kd + 辨识的粘滞摩擦)。
- git-lfs(数据 `*.pt` 存在 Git LFS 中)。

### 1. 安装

按 [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装(推荐 conda), 然后:

```bash
git lfs install
git clone -b keenon_l1 https://github.com/7huang/pace-sim2real.git
cd pace-sim2real
conda activate <isaaclab 环境>
python -m pip install -e source/pace_sim2real
```

确认数据是真实文件而不是 LFS 指针(约 258 KB; 若只有一百多字节, 执行 `git lfs pull`):

```bash
ls -l data/keenon_l1_hip_roll data/keenon_l1_hip_pitch
```

确认任务已注册(`scripts/list_envs.py` 只列 `Template-` 开头的任务, 查不到 L1):

```bash
python -c "from isaaclab.app import AppLauncher; app = AppLauncher(headless=True).app; import gymnasium as gym, pace_sim2real.tasks; print([k for k in gym.registry if 'KeenonL1' in k]); app.close()"
```

应输出 `Isaac-Pace-KeenonL1-HipRoll-v0` 与 `Isaac-Pace-KeenonL1-HipPitchBias-v0`。

### 2. 首次运行: 检查姿态

```bash
python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipRoll-v0 --num_envs 16
```

不加 `--headless`, 在窗口中确认: 机器人悬空、脚不触地; 两肘弯约 90°、腿自然下垂; 两个髋 roll 在做 chirp, 其余关节不动。看完 Ctrl+C。终端开头会打印 joint_order、27 个保持关节与各参数边界。

### 3. 髋 roll 辨识

```bash
python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipRoll-v0 --num_envs 2048 --headless
```

- `--num_envs` 即 CMA-ES 种群大小, 官方建议 512-2048 为常规辨识, 可先用 512 试跑估计耗时。
- 每一代都仿真完整的 12800 步(32 s); 最多 200 代, 或当代最好与最差得分相差 < 1% 时提前结束。

输出在 `logs/pace/keenon_l1_hip_roll/<YY_MM_DD_hh-mm-ss>/`:

| 文件 | 内容 |
|---|---|
| `mean_XXX.pt` | 每 10 代保存的参数均值(物理量) |
| `best_trajectory.pt` | 当前最优仿真轨迹 |
| `config.pt` | 边界与实测数据 |
| `result.json` | 结束时写出: 带关节名的参数、延迟(步与秒)、kd_scale |
| TensorBoard 日志 | 训练曲线 |

监控与检查:

```bash
tensorboard --logdir logs/pace/keenon_l1_hip_roll
python scripts/pace/plot_trajectory.py --robot_name keenon_l1_hip_roll --plot_trajectory --plot_score
```

| 检查项 | 正常 | 异常时 |
|---|---|---|
| score | 持续下降并收敛 | — |
| armature | 0.08-0.10 附近(电机表 0.095, 频域粗估 0.086) | 贴 0.02 或 0.2: 数据或模型有问题, 先停下分析 |
| 左右两侧 | armature、摩擦接近 | 差别大: 检查两侧数据 |
| 延迟 | 直方图收拢, 约 2 步(时间戳实测 4.7 ms) | 来回跳: 收窄延迟边界 |
| 粘滞摩擦 | > 0 | **贴在 0**: 有效阻尼比 0.9 x kd 还小, 把 `keenon_l1_pace_env_cfg.py` 的 `KD_SCALE` 改为 0.85 重跑 |
| 库伦摩擦 | 0.8 左右 | 贴上界 2: 放宽边界 |
| 零偏 | ±0.1 以内 | 贴 ±0.1: 检查采集零位 |
| plot_trajectory | 仿真与实测曲线基本重合 | — |

### 4. 髋 pitch 零偏

```bash
python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipPitchBias-v0 --num_envs 512 --headless
```

- 自动读取第 3 步最新一次运行中编号最大的 `mean_XXX.pt`: armature / 粘滞 / 库伦取左右平均、延迟取整数步, 全部固定(参数表中标"(固定)"), 只辨识 2 个零偏。
- 开头会打印"髋 pitch 固定参数来自 ...", 确认是第 3 步的结果文件而不是"频域粗估"。指定某次结果:
  ```bash
  PACE_L1_ROLL_RESULT=logs/pace/keenon_l1_hip_roll/<运行目录>/mean_XXX.pt python scripts/pace/fit_l1.py --task Isaac-Pace-KeenonL1-HipPitchBias-v0 --num_envs 512 --headless
  ```
- 只有 2 个有效参数, 收敛很快, 种群 512 足够。

### 5. 用到策略训练

把 `keenon_l1/training/keenon_l1_pace_training.py` 拷到训练工程; 训练环境中同样 `pip install -e source/pace_sim2real`(需要 `PaceDCMotorCfg`), 并同样使用 Isaac Sim 5.0。

```python
from keenon_l1_pace_training import load_pace_result, hip_actuator_cfg, HIP_EVENTS

roll = load_pace_result(".../logs/pace/keenon_l1_hip_roll/<运行目录>/result.json")
pitch = load_pace_result(".../logs/pace/keenon_l1_hip_pitch_bias/<运行目录>/result.json")

# 先把髋 pitch/roll 从原腿部执行器组中去掉(同一关节不能出现在两个组)
robot_cfg.actuators["hips"] = hip_actuator_cfg(roll, pitch, physics_dt=env_cfg.sim.dt, max_delay_steps=2)
# 域随机化加入训练的 EventCfg
for name, term in HIP_EVENTS.items():
    setattr(env_cfg.events, name, term)
```

- 延迟: PACE 以 2.5 ms 为一步, `hip_actuator_cfg` 按秒换算到训练 physics dt。使用 `hip_delay` 随机化时 `max_delay_steps` 不小于随机化上界换算的步数(dt = 0.005 时为 2)。
- 髋 pitch 随机化(借用参数, 放宽): armature 0.06-0.12, 库伦摩擦 0.5-1.2 Nm, 粘滞摩擦 0-0.5, 延迟 2.5-7.5 ms。
- 髋 roll 随机化: 文件中为占位值, 按 `result.json` 修改(建议 armature ±30%, 摩擦 ±40%)。
- 阻尼: 训练中执行器阻尼同样取 0.9 x kd(`load_pace_result` 会检查一致)。部署时策略下发的 kd 不变, 0.9 只是让仿真模拟真机偏小的实际阻尼。
- **零偏**: PaceDCMotor 只在 PD 中扣除零偏, 策略观测到的关节角是仿真真值, 不含零偏。按 PACE 约定, 真机编码器读数 = 真实角度 − 零偏。零偏较大(> 约 0.01 rad)时, 更稳妥的做法是在部署端修正(真实角度 = 读数 + 零偏), 训练中不加零偏; 零偏很小可忽略。

### 6. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `torch.load` 报 `invalid load key 'v'` | 数据是 LFS 指针: `git lfs install && git lfs pull` |
| `Setting joint viscous friction coefficients are not supported in Isaac Sim < 5.0` | Isaac Sim 版本过旧, 换 5.0 |
| URDF 导入失败或异常 | 设 `PACE_L1_URDF` 指向带网格的原 URDF(需 `../meshes/*.STL`) |
| 找不到 `chirp_data_info.json` | 数据目录须同时有 `chirp_data.pt` 与 `chirp_data_info.json` |
| `数据 dt=... 与仿真 dt=... 不一致` | 用 `to_pace.py --dt 0.0025` 重新转换 |
| 显存不足 | 减小 `--num_envs` |

## 数据

2026-10-10 的两组数据已用 `git add -f` 提交(`data/` 被 .gitignore 忽略, 新增或更新数据同样需要 `git add -f`; `*.pt` 走 Git LFS):

- `data/keenon_l1_hip_roll/`: leg_tool data 分支 `scripts/data/20261010_093547_m_hip_roll`(髋 roll 两腿 sym)
- `data/keenon_l1_hip_pitch/`: leg_tool data 分支 `scripts/data/20261010_093455_m_hip_pitch`(髋 pitch 两腿 anti)

新数据在 leg_tool 目录下从 pace_chirp 运行目录转换:

```bash
python scripts/to_pace.py <运行目录> --dt 0.0025 --joint-order left_hip_roll_joint,right_hip_roll_joint --out <pace-sim2real>/data/keenon_l1_hip_roll/chirp_data.pt
python scripts/to_pace.py <运行目录> --dt 0.0025 --joint-order left_hip_pitch_joint,right_hip_pitch_joint --out <pace-sim2real>/data/keenon_l1_hip_pitch/chirp_data.pt
```

## 状态

配置只在无 Isaac Lab 的环境中做过替身测试(读取 info、生成执行器与边界、读取髋 roll 结果), 尚未在 Isaac Lab 中运行。
