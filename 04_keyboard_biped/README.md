# 项目四：键盘驱动双足 MuJoCo 演示

本项目从零构建一台 10 关节简化双足机器人。基站位置和朝向由轨迹控制器更新，
髋、膝、踝、肩和肘在 MuJoCo 中由位置伺服驱动。

## 控制

| 按键 | 动作 |
| --- | --- |
| `W` / `↑` | 前进 |
| `S` / `↓` | 后退 |
| `A` / `←` | 左转 |
| `D` / `→` | 右转 |
| 空格 | 停止 |
| `Q` / `ESC` | 退出 |

## 运行

```powershell
python -m pip install -r requirements.txt
python main.py
python main.py --no-viewer
python main.py --test
python main.py --demo --output-dir results
```

一键运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_assignment.ps1
```

## 验证

- 8 项 pytest 全部通过。
- 0.3 m/s 运行 1 秒，位移正好 0.300 m。
- 先转 0.25 rad，再前进 0.3 m；实际位移方向与朝向完全一致。
- 零指令不会继续改变位置。
- 连续 100 次联合前进与转向后，全部关节状态保持有限值。

## 边界

机器人基座采用 mocap 引导，这是为了稳定展示键盘指令、关节回摆和场景运动。
它不是完整自由浮动双足动力学，也没有落脚点规划或 ZMP 平衡。实验验证的是
键位解码、朝向积分、关节伺服和渲染链路。
