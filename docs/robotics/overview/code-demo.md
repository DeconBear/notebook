---
title: "机器人学导论 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 机器人学导论 — demo.py 代码详解

<a href="/notebook/code/robotics/overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/robotics/overview/code
python demo.py
```

一张图 `overview_cspace.png`：左是 2R 工作空间采样 + 关节直线对应的手路径，右是 $\theta_1$-$\theta_2$ 里那条直线。杆长与运动学章相同（$1$ 与 $0.7$）。

## 代码逐段详解

### 第1步：FK 仍是两行余弦

与 [运动学](/robotics/kinematics/) 的 `fk` 相同。导论不讲逆解，只把「同一套角」映射到两个空间里看形状。

### 第2步：工作空间采样

```python
pts = np.array([fk(a, b) for a in th1s[::3] for b in th2s[::2]])
```

双重循环把构型网格送进 FK。这不是边界提取，只是「手能到的点云」，应落在半径 $[0.3,1.7]$ 的圆环带里。

### 第3步：构型空间直线

```python
th_path = th_a[None, :] + t[:, None] * (th_b - th_a)
ee = np.array([fk(*q) for q in th_path])
```

- **`th_a[None, :]`**：`(2,)` → `(1,2)`，与 `t[:, None]` 的 `(n,1)` 广播成 `(n,2)`。
- **手路径 `ee` 是弯的**：这是本章唯一要看见的事实。下一章轨迹规划会再比较线性 vs 三次，手路径仍然弯。

左图两套折线是起终点姿态；圆点和方点标手的起终。`set_aspect('equal')` 两边都要，否则圆环被压扁、$\theta$ 平面比例失真。

## 源码位置

`docs/robotics/overview/code/demo.py`
