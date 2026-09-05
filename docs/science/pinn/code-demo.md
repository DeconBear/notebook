---
title: "as02 PINN — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# as02 PINN — demo.py 代码详解

<a href="/notebook/code/science/pinn/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/science/pinn/code
python demo.py
```

依赖 PyTorch（CPU 即可，问题规模极小，几秒内训练完成）。

## 代码逐段详解

### 第1步：问题定义（复用 as01 的制造解）

```python
def u_true(x):
    return np.sin(np.pi * x)

def f_source(x):
    return (np.pi ** 2) * torch.sin(np.pi * x)
```

与 as01 完全相同的一维 Poisson 问题：$-u''(x)=f(x)$，$u(0)=u(1)=0$，真解 $u(x)=\sin(\pi x)$。注意 `f_source` 这里用的是 `torch.sin`，因为它要参与自动微分计算图（输入 `x` 是需要求导的 tensor）。

### 第2步：PINN 网络结构

```python
class PINN(nn.Module):
    def __init__(self, hidden_dim=20, n_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
```

结构是 `1 → 20 → 20 → 20 → 1` 的全连接网络，激活函数用 **tanh**（而非 ReLU）——因为 PINN 需要对网络输出求二阶导数，ReLU 的二阶导数几乎处处为 0，无法提供有效的训练信号；tanh 处处光滑可微。配合 Xavier 初始化，这是 tanh 网络的标准搭配。

### 第3步：用自动微分计算 PDE 残差 —— 全章最核心的函数

```python
def compute_pde_residual(model, x):
    u = model(x)
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]
    residual = -d2u_dx2 - f_source(x)
    return residual
```

**关键点**：
- `torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]` 计算 $\partial u/\partial x$。`grad_outputs=torch.ones_like(u)` 是因为 `u` 是一个向量（每个配点一个输出），我们要对每个分量分别求导（等价于逐元素的雅可比对角线）。
- **`create_graph=True` 是让二阶导数可行的关键**：它让"求一阶导数"这个操作本身也被记录进计算图，于是我们可以对 `du_dx` **再次**调用 `torch.autograd.grad`，得到二阶导数 `d2u_dx2`，并且这个二阶导数仍然带着完整的计算图，可以继续 `.backward()` 更新网络权重。
- 组装出残差 `r(x) = -u_theta''(x) - f(x)`，与 as01 中有限差分版本的定义完全一致，只是这里的导数是**解析精确**的。

### 第4步：训练循环 —— 复合损失函数

```python
def train_pinn(n_collocation=50, n_epochs=3000, lr=1e-3, lambda_bc=10.0):
    x_f = torch.linspace(0, 1, n_collocation, device=DEVICE).reshape(-1, 1)
    x_f.requires_grad_(True)   # 必须开启梯度追踪，否则无法对 x 求导

    x_b = torch.tensor([[0.0], [1.0]], device=DEVICE)
    u_b_true = torch.tensor([[0.0], [0.0]], device=DEVICE)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        residual = compute_pde_residual(model, x_f)
        loss_pde = torch.mean(residual ** 2)

        u_b_pred = model(x_b)
        loss_bc = torch.mean((u_b_pred - u_b_true) ** 2)

        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        optimizer.step()
```

**易错点提醒**：`x_f.requires_grad_(True)` 必须在构造张量之后显式设置——如果忘记这一步，`torch.autograd.grad` 会因为 `x` 不在计算图的"叶子节点"上而报错。

**边界条件权重 `lambda_bc=10.0`**：本问题只有 2 个边界点，而配点有 50 个。如果 `lambda_bc=1`，边界条件损失容易被"稀释"，网络可能在内部拟合得不错但边界处明显偏离 0。把权重放大 10 倍，是让优化器更"重视"这两个边界点的常见技巧。

### 第5步：可视化 —— 预测解 vs 真解、损失曲线

```python
l2_rel_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
```

用相对 $L_2$ 误差量化最终精度（demo 运行后会打印，通常在 $10^{-3}$ 量级）。损失曲线图（`pinn_loss.png`）用对数坐标同时展示总损失、PDE 残差损失、边界条件损失的下降轨迹。

### 第6步：PINN 架构示意图

用 `FancyBboxPatch`/`FancyArrowPatch` 手绘架构图（`as02-01-pinn-architecture.png`）：输入 → MLP → 输出，输出分两路——一路自动微分两次组装 PDE 残差损失，一路直接用边界点评估边界条件损失，两路损失加权汇总后反向传播。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| collocation points | 定义域内采样的点，用于评估 PDE 残差（不是"训练数据"，因为没有标签） | `x_f` |
| create_graph=True | 让求导操作本身可被再次求导，是计算二阶导数的关键 | `compute_pde_residual()` |
| tanh 激活 | 处处光滑可微，二阶导数不退化为 0，适合 PINN | `PINN.__init__()` |
| 复合损失 | $L = L_{\text{pde}} + \lambda_{\text{bc}} L_{\text{bc}}$ | `train_pinn()` |
| 相对 L2 误差 | $\|\hat u - u\|_2 / \|u\|_2$，评估整体拟合精度 | `plot_solution_comparison()` |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/science/pinn/code/demo.py`
