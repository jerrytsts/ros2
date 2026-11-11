# 项目二：二维双足行走与 ZMP 裕度

本项目用线性倒立摆描述前后方向的质心动力学，按交替落脚点计算支撑区间，
再通过发散分量反馈选择 ZMP。代码结构、实验调度、指标和报告均为独立实现。

## 模型

- 质心高度固定为 0.80 m。
- `ω=sqrt(g/zc)`。
- 给定 ZMP 后使用双曲函数做恒 ZMP 解析推进。
- 单脚支撑时 ZMP 参考在脚掌中心；双脚支撑时使用 smoothstep 过渡。
- 反馈律：`ZMP = 参考值 + K·(DCM-DCM参考)`。
- ZMP 始终裁剪在当前支撑区间。

## 运行

```powershell
python -m pip install -r requirements.txt
python main.py test
python main.py run
```

一键复现：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_assignment.ps1
```

## 当前结果

- 11 项测试通过。
- 默认 `K=1.05` 时，7.118 秒前进 5.122 m。
- 平均速度 0.720 m/s，目标平均速度约 0.739 m/s，达到目标的 97.3%。
- 全部采样 ZMP 都在支撑范围内，最小裕度为 0。
- `K=0` 和 `K=0.9` 出现指数发散；`K≥1.05` 的完整步态能够收敛到合理行程。
- 周期初值测试和开环不稳定测试确认：周期解存在，但误差会快速放大。

输出位于 `results/`：

- `gain_sweep.csv`
- `walking_profile.png`
- `summary.json`

## 边界

这是前后方向的单质点 LIPM，不包含左右踝力矩、脚掌转动、关节极限和柔顺接触。
它验证的是步态规划和 DCM 反馈，不代替完整三维机器人动力学。
