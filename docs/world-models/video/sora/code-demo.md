---
title: "wm07 视频生成式世界模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm07 视频生成式世界模型 — demo.py 代码详解

<a href="/notebook/code/world-models/video/sora/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/video/sora/code
python demo.py
```

## 代码逐段详解

### 第1步：架构示意图

`plot_video_wm_architecture()` 画出"文本条件 → 时空潜空间 → 扩散/自回归生成 → 视频"流水线，并标注四大物理一致性瓶颈，输出 `wm07-01-video-wm.png`。

### 第2步：合成移动光斑

```python
def render_blob(pos, h=16, w=16, sigma=1.2):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return np.exp(-((yy - pos[0])**2 + (xx - pos[1])**2) / (2 * sigma**2))
```

每条轨迹：随机起点 + 随机速度，碰壁反弹。这是一个有解析物理的极简"世界"，适合检验预测器是否学到了运动规律。

### 第3步：CNN 下一帧预测器

把过去 `HIST=3` 帧沿通道维拼接，经 3 层卷积输出下一帧。损失是像素 MSE。玩具规模下几秒就能收敛。

### 第4步：开环滚动预测（核心实验）

```python
frames = [video[0], video[1], video[2]]   # 真实帧启动
for _ in range(n_roll):
    pred = model(stack(frames[-3:]))
    frames.append(pred)                    # 把预测喂回去！
```

与"教师强制"（每步都用真实历史）不同，开环把预测误差带回输入，导致误差随时间放大——这正是视频生成模型做长时程世界推演时的核心脆弱点。

### 第5步：质心轨迹与误差增长

用亮度加权平均估计光斑质心，对比真实/预测轨迹，并绘制帧 MSE、质心欧氏误差随开环步数的增长曲线。

### 关键概念速查表

| 概念 | 一句话 | 代码位置 |
|------|--------|---------|
| 开环滚动 | 预测结果喂回模型连续推演 | `open_loop_rollout` |
| 教师强制 | 每步用真实历史（本 demo 训练时使用） | `build_supervised_pairs` |
| 质心误差 | 物理位置层面的偏离度量 | `blob_centroid` |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/video/sora/code/demo.py`
