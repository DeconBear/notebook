---
title: "s08 优化器：从SGD到Adam — demo.py"
---

# s08 优化器：从SGD到Adam — demo.py

<a href="../code/s08_optimizers_sgd_to_adam/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

在狭长峡谷形二维损失地形（L(theta) = 0.5 * (20*theta1^2 + theta2^2)，条件数 kappa=20）上可视化对比 SGD、Momentum、RMSProp、Adam 四种优化器的优化轨迹、损失下降曲线，以及超参数游乐场（不同学习率的效果）。额外展示 Adam 对梯度噪声的鲁棒性远优于 SGD。

## Run It

```bash
cd s08_optimizers_sgd_to_adam/code
python demo.py
```

## Key Output

- 损失地形：二维椭球面，theta1方向曲率20（陡峭），theta2方向曲率1（平缓），条件数=20
- 初始点：(3.0, 3.0)，最优解：(0, 0)，最优损失=0
- SGD：在陡峭方向来回震荡（之字形），收敛慢
- Adam：结合Momentum和RMSProp优势，收敛最快最稳
- 噪声鲁棒性：梯度加噪声后，Adam仍能平稳收敛，SGD剧烈抖动

<<< @/snippets/s08_optimizers_sgd_to_adam/demo.py
