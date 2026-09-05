---
title: "导论：从膜电位到 NeuroAI — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 导论：从膜电位到 NeuroAI — demo.py 代码详解

<a href="/notebook/code/neuro/overview/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/neuro/overview/code
python demo.py
```

CPU、matplotlib 即可。输出一张图 `neuro_scale_ladder.png`：六级「尺度梯子」，左列生物尺度、中列典型问题、右列本教程对应模型。没有微分方程、没有网络训练——导论只把后续章节钉在一张地图上。

## 代码逐段详解

### 第1步：导入 — 为什么需要 `FancyBboxPatch`

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
```

- **`FancyBboxPatch`**：圆角矩形补丁。普通 `Rectangle` 是尖角；梯子每一格是卡片，圆角更像「板块」而不是坐标轴上的 bar。
- 中文字体列表 + `axes.unicode_minus = False`：标题和格子里全是中文，缺字体就方块。
- **本文件没有 NumPy**：没有数组运算。`LEVELS` 是纯 Python 元组列表。

路径仍用脚本相对 `images/`：

```python
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
```

---

### 第2步：`LEVELS` — 数据与画图分离

```python
LEVELS = [
    ('离子 / 通道', 'Nernst、门控', 'HH 电流项'),
    ('单细胞膜', '静息、动作电位', 'HH / LIF'),
    ('突触', 'EPSP / IPSP、可塑性', 'STDP'),
    ('回路', 'E–I、感受野', '方向选择性 · raster'),
    ('结构', '谁连谁', '连接组 / SONATA-lite'),
    ('与 AI', '启发 / 对齐 / 约束', 'NeuroAI'),
]
```

每个三元组：`(生物尺度, 典型问题, 本教程模型)`。改文案只改这里，循环里的坐标公式不用动。自上而下从离子走到 NeuroAI，和后续章节顺序一致：`hh-lif` → `neuron`/`stdp` → `circuits` → `connectomics` → `neuroai`。

**为什么用元组不是字典？** 只要三个位置固定的短字符串，元组更短；循环里 `bio, q, model` 解包即可。

---

### 第3步：坐标系关掉轴，自己当画布

```python
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
```

- **`figsize=(9.5, 5.2)`**：英寸。宽一点才能放下三列卡片。
- **`set_xlim` / `set_ylim`**：数据坐标。卡片的 `(x, y, 宽, 高)` 都按这套尺子写。
- **`axis('off')`**：不要刻度、不要边框。这是示意图不是函数图像。

---

### 第4步：循环里三列卡片 + 竖向箭头

```python
colors = ['#d4e6f1', '#fdebd0', '#d5f5e3', '#fadbd8', '#e8daef', '#d6eaf8']
for i, ((bio, q, model), c) in enumerate(zip(LEVELS, colors)):
    y = 5.6 - i * 0.9
```

- **`zip(LEVELS, colors)`**：第 $i$ 级配第 $i$ 个颜色。两列等长，多出来的会被丢掉。
- **嵌套解包 `((bio, q, model), c)`**：`zip` 给出 `(三元组, 颜色)`，再把三元组拆开。
- **`y = 5.6 - i * 0.9`**：第 0 级在上。`0.9` 是行距（卡片高 0.72 + 间隙）。$i$ 增大 $y$ 减小，符合「往下走一层」。

```python
ax.add_patch(FancyBboxPatch((0.4, y), 3.2, 0.72, boxstyle='round,pad=0.02',
                            facecolor=c, edgecolor='#333', linewidth=1.2))
ax.text(2.0, y + 0.36, bio, ha='center', va='center', ...)
```

- **`FancyBboxPatch((x, y), width, height)`**：左下角在 `(0.4, y)`，宽 3.2、高 0.72。
- **`boxstyle='round,pad=0.02'`**：圆角；`pad` 控制圆角半径量级。
- **`ax.text(..., ha='center', va='center')`**：文字锚在格子中心。$x=2.0$ 是左列中线（$0.4+3.2/2$），$y+0.36$ 是格子垂直中点（高 0.72 的一半）。
- 中列白底、右列浅灰描蓝边：视觉上「问题」中性、「模型」是教程入口。

```python
if i < len(LEVELS) - 1:
    ax.annotate('', xy=(2.0, y - 0.08), xytext=(2.0, y - 0.16),
                arrowprops=dict(arrowstyle='-', color='#555', lw=1.2))
```

- **最后一级不要往下指**：`i < len-1`。
- **`annotate` 的 `xy` 是箭头尖、`xytext` 是尾**。这里 `arrowstyle='-'` 其实是短线段，不是三角箭头，只在左列卡片之间留一条「梯子竖杆」。坐标都在左列中轴 $x=2.0$。

三列表头用 `ax.text(..., 6.55, ...)` 写在最顶上一行之上。

---

### 第5步：保存

```python
fig.tight_layout()
out = os.path.join(_IMAGES_DIR, 'neuro_scale_ladder.png')
fig.savefig(out, dpi=140)
plt.close(fig)
```

- **`tight_layout`**：压缩边距，标题不被裁。
- **`dpi=140`**：屏幕阅读够清，文件不太大。
- **`plt.close(fig)`**：关掉图窗，避免在无显示器的脚本里堆积后端对象。

入口是 `if __name__ == '__main__': main()`，没有别的函数：导论脚本 = 一张图。

---

### 关键概念速查表

| 概念 | 直觉 | 代码 |
|------|------|------|
| 尺度梯子 | 离子→细胞→突触→回路→结构→AI | `LEVELS` 六行 |
| 圆角卡片 | 示意图不是统计图 | `FancyBboxPatch` |
| `zip` | 行与颜色配对 | `zip(LEVELS, colors)` |
| `y = 5.6 - i*0.9` | 自上而下 | 行距 |
| `ha/va='center'` | 字在格子正中 | `ax.text` |
| `axis('off')` | 关掉坐标轴 | 画布模式 |
| `annotate` | 级间短线 | `arrowstyle='-'` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/neuro/overview/code/demo.py`
