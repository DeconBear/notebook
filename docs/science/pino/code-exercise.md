---
title: "as04 PINO — exercise.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as04 PINO — exercise.py 练习指南

<a href="/notebook/code/science/pino/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

1. 实现有限体积形式的 PDE 残差损失  
2. 拼出 PINO 总损失 = 数据损失 + $\lambda\cdot$PDE 残差 + 边界  
3. 验证：同样 2 个标注点时，带物理约束的模型外推更准

## 任务清单

### 任务1：`pde_residual_loss`

按提示用 `k_half`、左右通量差构造

$$
R = -\frac{F_{i+1/2}-F_{i-1/2}}{\Delta x} - f_i
$$

并对内部点取均方。用数值精确解代入时，残差应 $<10^{-8}$。

### 任务2：`pino_total_loss`

- 数据项：只在 `labeled_idx` 上算 MSE  
- 物理项：对**全部** batch 调用任务1  
- 边界项：$\hat{u}(:,0)^2 + \hat{u}(:,-1)^2$

## 验证标准

```bash
cd docs/science/pino/code
python exercise.py
```

若实现正确，将看到「✓ PINO 物理残差实现正确」，并生成 `exercise_pino_extrapolation.png`。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/pino/code/exercise.py`
