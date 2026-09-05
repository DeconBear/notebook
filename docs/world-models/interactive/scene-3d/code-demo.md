---
title: "交互/3D — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 交互 / 3D — demo.py 代码详解

<a href="/notebook/code/world-models/interactive/scene-3d/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/interactive/scene-3d/code
python demo.py
```

秒级。5×5 格子：左边用显式键位（上下左右）滚动；右边用状态差分的 k-means 当「潜动作」，再滚同一条意图。不是 3D 网格或 NeRF，只演示 **动作接口可以是学出来的**。更完整的无监督离散动作见 [Genie](/world-models/interactive/genie/code-demo)。

## 代码逐段详解

### 第1步：显式动作表

```python
ACTIONS = {
    0: np.array([-1, 0]),  # 行减一：向上（图像坐标行向下为正，这里行减小是「往网格上方」）
    1: np.array([1, 0]),
    2: np.array([0, -1]),
    3: np.array([0, 1]),
}
```

整数 id → 二维位移。`step` 里 `np.clip(..., 0, H-1)`：撞墙停住，不会包绕。**语法字典映射**：规划器输出 0–3，环境查表。这是「手柄按键」接口。

---

### 第2步：收集演示轨迹

```python
pos = np.array([np.random.randint(H), np.random.randint(W)])
a = np.random.randint(4)
pos = step(pos, a)
seq.append(pos.copy())
```

**`.copy()`**：`pos` 会被原地改（`clip` 返回新数组其实已是新对象，但养成习惯：list 里若存同一引用，后面一改全改）。每条轨迹长度 `T=12`，共 200 条，给聚类足够的差分样本。

---

### 第3步：从 $\Delta s$ 发现潜动作（教学版 k-means）

```python
d = seq[1:] - seq[:-1]
deltas = np.concatenate(deltas, axis=0)
centers = deltas[np.random.choice(len(deltas), k, replace=False)]
dist = ((deltas[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
lab = dist.argmin(axis=1)
```

- **`seq[1:]-seq[:-1]`**：相邻状态差。无动作标签，只看「格子怎么动了」。撞墙时差分为 0，会出现「原地」簇，这是边界代价。
- **`[:, None, :]`**：插入长度为 1 的轴，形状变成 `(N, 1, 2)`，与 `centers[None,:,:]` 的 `(1, k, 2)` 广播，得到 `(N, k, 2)` 再 `sum(-1)` → `(N, k)` 距离。这是不调用 `sklearn` 的成对欧氏距离。
- **`argmin(axis=1)`**：每个差分属于最近中心。
- 迭代 20 次：有样本的簇更新为均值；空簇保持原中心（`if np.any(lab==j)`）。

理想情况下 4 个中心接近 `(±1,0)`、`(0,±1)`。打印 `np.round(centers, 3)` 自己核对。

---

### 第4步：把显式动作映射到最近潜码再滚动

```python
for a in explicit_acts:
    d = ACTIONS[a].astype(float)
    lid = np.argmin(((centers - d) ** 2).sum(axis=1))
    latent_ids.append(lid)
```

规划仍用人类键位；执行潜动作时查「这个位移最像哪个簇」。`rollout_latent`：

```python
pos = np.clip(pos + centers[int(lid)], 0, H - 1)
```

中心是浮点均值，路径可能在格子之间——图上仍能看出与显式路径同走向。这就是路径二的直觉：交互接口不必是「上/下/左/右」这个标签集，可以是视频变化里聚出来的码。

**语法 `imshow` + `plot(path[:,1], path[:,0])`**：图像坐标 x 是列 = `path[:,1]`，y 是行 = `path[:,0]`，和 `ACTIONS` 的 `[行, 列]` 一致。

---

### 关键概念速查表

| 概念 | 落地 |
|------|------|
| 显式动作 | `ACTIONS` 整数 id |
| 潜动作 | k-means on $\Delta s$ |
| 广播距离 | `[:,None,:]` |
| `.copy()` | 避免 list 存同一引用 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/interactive/scene-3d/code/demo.py`
