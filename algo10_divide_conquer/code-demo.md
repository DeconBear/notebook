---
title: "algo10 递归、分治与二分 — demo.py"
---

# algo10 递归、分治与二分 — demo.py 代码详解

<a href="../code/algo10_divide_conquer/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd algo10_divide_conquer/code
python demo.py
```

## 代码逐段详解

### 第1步：归并排序 — 分治法的标准模板

归并排序完美体现了分治三阶段——分解(Divide)、解决(Conquer)、合并(Combine)。

```python
def merge_sort(arr, left=0, right=None):
    if left >= right:
        return [arr[left]]  # 基线：单元素数组天然有序
    mid = (left + right) // 2
    left_sorted = merge_sort(arr, left, mid)     # 解决左半
    right_sorted = merge_sort(arr, mid + 1, right)  # 解决右半
    return merge(left_sorted, right_sorted)      # 合并
```

`merge()` 使用双指针法合并两个有序数组：两个指针 i 和 j 分别指向左右子数组的起始位置，每次比较 `left[i]` 和 `right[j]`，取较小的放入结果，并将对应指针后移。当一个子数组耗尽后，直接将另一子数组的剩余部分追加。

**复杂度分析**：$T(n) = 2T(n/2) + O(n)$，由主定理得 $T(n) = O(n \log n)$。

### 第2步：逆序对计数 — 在合并过程中"顺手"统计

逆序对的数量衡量了数组的"混乱程度"。

```python
def merge_and_count(left, right):
    result = [] ; i = j = 0 ; inv_count = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j])
            inv_count += len(left) - i  # ← 关键行：横跨逆序对！
            j += 1
    ...
```

**为什么 `len(left) - i`？** 当 `right[j]` 被放入结果时，说明 `right[j] < left[i]`。由于 left 已排序，`left[i:]` 中的所有元素都大于 `right[j]`，每个都与 `right[j]` 构成一个逆序对。因此累加 `len(left) - i`。

**运行示例**：对 `[2, 4, 1, 3, 5]`：
- 分解为 `[2, 4]` 和 `[1, 3, 5]`
- `[2, 4]` 内部：0 个逆序对
- `[1, 3, 5]` 内部：0 个逆序对
- 合并时：`right[0]=1` 比 `left[0]=2` 小 → 横跨 2 个逆序对 `(2,1), (4,1)`；然后 `4 > 3` → 横跨 1 个 `(4,3)`；共 3 个。

### 第3步：快速排序 — 三种 pivot 策略

快速排序的核心在 `partition()` 上——选择 pivot 并将数组分为"小于 pivot"和"大于 pivot"两部分。

#### Lomuto 方案

```python
def partition_lomuto(arr, lo, hi):
    pivot = arr[hi]  # 以最后一个元素为 pivot
    i = lo - 1       # "小元素边界"
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # 把小的往前换
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]  # pivot 归位
    return i + 1
```

i 指针始终指向"最后一个 ≤ pivot 的元素"。当 j 扫描到一个 ≤ pivot 的元素时，i 向右移一位（腾出空间），然后交换，将该元素放入"小元素区"。

#### 随机化快排

最关键的改进：随机选择 pivot 来避免最坏情况 $O(n^2)$。

```python
pivot_idx = random.randint(lo, hi)
arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]  # 随机 swap 到末尾
```

#### Hoare 方案

从两端向中间双向扫描，比 Lomuto 更高效（约少 30% 的交换次数）。

### 第4步：二分搜索变体

```python
def lower_bound(arr, target):
    lo, hi = 0, len(arr)  # 注意 hi = len(arr)，不是 len(arr)-1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

**关键区别**：
- `hi = len(arr)` 而非 `len(arr)-1`：允许返回 `len(arr)`（所有元素都 < target）
- `lo < hi` 而非 `lo <= hi`：因为 hi 是半开区间端点
- `arr[mid] < target` 而非 `<=`：`<` 时 `lo = mid+1`，`>=` 时 `hi = mid`

**实例**：`lower_bound([1,2,4,4,4,6,8,8,10], 4)`：
- `lo=0, hi=9, mid=4→arr[4]=4, hi=4`
- `lo=0, hi=4, mid=2→arr[2]=4, hi=2`
- `lo=0, hi=2, mid=1→arr[1]=2<4, lo=2`
- `lo=2, hi=2` → 返回 2（第一个 4 的位置）

### 第5步：三分搜索

用于寻找单峰函数的极值点。每次取两个三等分点，利用函数值的比较来缩小区间：

- `f(m1) < f(m2)` → 极值在左半区 → 缩小到 `[lo, m2]`
- `f(m1) > f(m2)` → 极值在右半区 → 缩小到 `[m1, hi]`
- `f(m1) = f(m2)` → 极值在中间 → 缩小到 `[m1, m2]`

每次迭代区间缩小约 33%，$O(\log_{1.5}(1/\varepsilon))$ 次迭代即可达到精度要求。

### 第6步：牛顿迭代法

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**几何解释**：在点 $(x_n, f(x_n))$ 处作切线，切线的方程为 $y = f(x_n) + f'(x_n)(x - x_n)$。令 $y=0$ 解得 x 截距，即为 $x_{n+1}$。

**演示**：求 $x^3 - 2x - 5 = 0$ 的根。初始 $x_0 = 2.5$，3-4 次迭代即可收敛到机器精度。

## 关键概念速查表

| 概念 | 要点 | 代码位置 |
|------|------|---------|
| 分治三步骤 | Divide → Conquer → Combine | `merge_sort()` |
| 逆序对计数 | 合并时累加左半剩余元素数 | `merge_and_count()` |
| Lomuto 划分 | pivot=最后一个，i 维护小元素边界 | `partition_lomuto()` |
| 随机化快排 | 随机选 pivot 交换到末尾 | `quicksort_randomized()` |
| 二分下界 | 第一个 ≥ x 的位置，hi=len(arr) | `lower_bound()` |
| 二分答案 | 对单调性答案空间二分 | `binary_search_answer()` |
| 三分搜索 | 两个三等分点比较，缩小区间 | `ternary_search()` |
| 牛顿法 | 切线迭代，二次收敛 | `newtons_method()` |

## 完整代码

<<< @/snippets/algo10_divide_conquer/demo.py
