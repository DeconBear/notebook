---
title: "s03 逻辑回归 — demo.py"
---

# s03 逻辑回归 — demo.py

<a href="../code/s03_logistic_regression/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现二分类逻辑回归（Sigmoid + 交叉熵 + 梯度下降）和多分类 Softmax 回归，使用 Iris 数据集进行训练和评估。展示 Sigmoid 函数形态、二分类决策边界与概率热力图、多分类决策区域、以及训练损失收敛曲线。

## Run It

```bash
cd s03_logistic_regression/code
python demo.py
```

## Key Output

- 数据集：Iris 数据集，150样本，4特征，3类别
- 二分类（类别0 vs 1）：测试集准确率 100%
- 多分类（3类 Softmax）：测试集准确率约 80%
- 关键数学事实：Sigmoid + 交叉熵的梯度 = y_pred - y，极其简洁

<<< @/snippets/s03_logistic_regression/demo.py
