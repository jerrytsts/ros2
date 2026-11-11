# 项目五：七轴桌面机械臂键盘控制

模型是一个原创的七轴桌面机械臂，附两指夹爪。7 个转动关节由 MuJoCo 位置伺服
驱动，所有关节都有范围限制。项目不依赖 MoveIt、ROS 或外部模型仓库。

## 按键

| 按键 | 动作 |
| --- | --- |
| `1`～`7` | 对应关节正向转动 0.1 rad |
| `Shift+1`～`Shift+7` | 对应关节反向转动 0.1 rad |
| `R` | 回到初始姿态 |
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

- 6 项测试全部通过。
- 模型包含 9 个自由度、9 个执行器，其中 7 个为转动关节。
- 初始位姿最大跟踪误差约 0.043 rad。
- 关节 2 改变 0.18 rad 后，稳态误差约 0.028 rad。
- 离屏渲染为 480×360 RGB。
- 自动演示让多个关节正反向运动，最后复位。

## 边界

这是作业用的简化机械臂动力学模型，不是 Franka Panda 官方参数或碰撞规划器。
它验证键位映射、增量目标、限位、位置伺服与可视化，不验证抓取规划、碰撞检测
或真实电机特性。
