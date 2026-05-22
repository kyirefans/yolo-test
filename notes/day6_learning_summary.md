# Day6 学习总结：PR Curve、AP 与 mAP

## 1. 今天的核心问题

单一 `Precision` / `Recall` 不能完整评价检测模型，因为它们会受到 `confidence threshold` 的影响。

同一个模型在不同置信度阈值下会表现出不同的 Precision 和 Recall：

- 高 `conf`：模型更保守，预测框更少，Precision 往往更高，Recall 往往更低。
- 低 `conf`：模型更激进，预测框更多，Recall 往往更高，Precision 往往更低。

所以，不能只用某一个阈值下的 Precision / Recall 判断模型整体好坏。

## 2. 三种 threshold 的区别

| 阈值 | 比较对象 | 作用阶段 | 作用 |
| --- | --- | --- | --- |
| `confidence threshold` | 预测框自己 | 推理过滤 | 决定低置信度预测框是否保留 |
| `evaluation IoU threshold` | 预测框 vs 真实框 | 指标评估 | 决定预测框能否算 TP |
| `NMS IoU threshold` | 预测框 vs 预测框 | 后处理 | 去掉重复预测框 |

一个预测框 confidence 很高，不代表它一定是 TP。  
如果它和真实框的 IoU 小于 evaluation IoU threshold，它仍然是 FP。

例如：

```text
confidence = 0.92
Pred-GT IoU = 0.31
evaluation IoU threshold = 0.5
```

因为 `0.31 < 0.5`，所以这个预测框是 FP。

## 3. PR Curve 如何构造

PR Curve 不是简单手动试几个 `conf` 值画出来的线。

更标准的构造方式是：

1. 选定一个类别，例如 `person`。
2. 收集该类别所有预测框。
3. 按 confidence 从高到低排序。
4. 从最高 confidence 的预测框开始，一个一个加入统计。
5. 每加入一个预测框，就更新累计 TP、累计 FP。
6. 计算当前 Precision 和 Recall。

公式：

```text
Precision = TP / (TP + FP)
Recall    = TP / GT 总数
```

加入 FP 时：

```text
TP 不变
FP 增加
Recall 不变
Precision 下降
```

加入 TP 时：

```text
TP 增加
Recall 上升
Precision 不一定上升
```

## 4. 手算例子

设：

```text
GT 总数 = 4

按 confidence 排序后的预测结果：
1. TP
2. FP
3. TP
4. FP
5. TP
```

逐步计算：

| 步骤 | 判断 | 累计 TP | 累计 FP | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | TP | 1 | 0 | 1 / 1 = 1.000 | 1 / 4 = 0.250 |
| 2 | FP | 1 | 1 | 1 / 2 = 0.500 | 1 / 4 = 0.250 |
| 3 | TP | 2 | 1 | 2 / 3 = 0.667 | 2 / 4 = 0.500 |
| 4 | FP | 2 | 2 | 2 / 4 = 0.500 | 2 / 4 = 0.500 |
| 5 | TP | 3 | 2 | 3 / 5 = 0.600 | 3 / 4 = 0.750 |

对应 PR 点：

```text
(Recall=0.25, Precision=1.000)
(Recall=0.25, Precision=0.500)
(Recall=0.50, Precision=0.667)
(Recall=0.50, Precision=0.500)
(Recall=0.75, Precision=0.600)
```

## 5. AP 的含义

AP 是 Average Precision。

它可以理解为：

```text
AP = PR 曲线的综合面积
```

但更准确地说：

```text
AP 衡量模型在 Recall 不断提高的过程中，Precision 能否保持较高水平。
```

AP 不是某一个 `conf` 下的 Precision，也不是 `conf / recall / precision` 三维图形的面积。

`confidence` 的作用是排序：

```text
confidence 决定预测框从高到低的统计顺序
排序顺序产生一串 Precision / Recall 点
这些点形成二维 PR Curve
AP 是二维 PR Curve 的综合结果
```

## 6. 为什么高置信度 FP 很伤 AP

如果一个 FP 的 confidence 很高，它会排在统计序列前面。

这会导致：

- 一开始 Precision 就被拉低。
- PR 曲线前段表现变差。
- AP 降低。

所以 AP 不只关心错了多少，也关心模型是否把错误预测排得很靠前。

例子：

```text
模型 A：TP, TP, TP, FP, FP
模型 B：FP, FP, TP, TP, TP
```

虽然两个模型最后都找到了 3 个 TP，并且都有 2 个 FP，但模型 A 的 AP 更高。  
因为模型 A 把高置信度位置留给了正确预测，而模型 B 在最前面就出现了高置信度 FP。

## 7. AP@0.5、AP@0.75 和 mAP@0.5:0.95

AP 的计算必须先确定 evaluation IoU threshold。

例如：

```text
Pred-GT IoU = 0.62
类别正确
```

在不同评价条件下：

| 指标 | 判断 |
| --- | --- |
| `AP@0.5` | TP，因为 `0.62 >= 0.5` |
| `AP@0.75` | FP，因为 `0.62 < 0.75` |

含义：

- `AP@0.5`：IoU 阈值为 0.5，定位要求较宽松，更关注是否大致找到目标。
- `AP@0.75`：IoU 阈值为 0.75，定位要求更严格，更关注框的位置是否准确。
- `mAP@0.5:0.95`：在 `0.50, 0.55, ..., 0.95` 多个 IoU 阈值下计算并平均，更严格、更综合。

## 8. AP 和 mAP 的关系

AP 是单类别指标，例如：

```text
person 的 AP
car 的 AP
chair 的 AP
```

mAP 是多个类别 AP 的平均：

```text
mAP = mean Average Precision
```

所以：

```text
mAP50     = 所有类别在 IoU=0.5 下的 AP 平均
mAP75     = 所有类别在 IoU=0.75 下的 AP 平均
mAP50-95  = 所有类别在 IoU=0.50 到 0.95 多个阈值下的 AP 平均
```

## 9. 对应到当前代码

在 `scripts/validate_coco128.py` 中：

```python
results = model.val(
    data="coco128.yaml",
    imgsz=640,
    plots=True,
    verbose=True,
)
```

这一步执行 COCO128 验证。

代码中打印的指标：

```python
print(f"mAP50-95: {results.box.map:.4f}")
print(f"mAP50:    {results.box.map50:.4f}")
print(f"mAP75:    {results.box.map75:.4f}")
```

对应关系：

```text
results.box.map50 -> mAP@0.5
results.box.map75 -> mAP@0.75
results.box.map   -> mAP@0.5:0.95
```

## 10. 当前实验结果解释

当前 `experiments/day5_coco128_validation/coco128_validation_summary.json` 中的结果是：

```text
mAP50    = 0.6699
mAP75    = 0.5390
mAP50-95 = 0.5026
```

可以解释为：

```text
YOLO11n 在 COCO128 上已经具备一定检测能力。
mAP50 高于 mAP75，说明模型在宽松 IoU 条件下能找到不少目标，
但当定位要求变严格时，部分预测框的位置不够准。

mAP50 高于 mAP50-95，说明随着 IoU 阈值从 0.5 提高到 0.95，
模型性能逐步下降。这是目标检测中常见现象，也说明定位精度还有提升空间。
```

## 11. 今天必须记住的话

1. 单点 Precision / Recall 只代表某一个 confidence threshold 下的工作点。
2. PR Curve 描述的是模型在不同排序截断点下的 Precision-Recall 关系。
3. AP 是单类别 PR 曲线的综合结果。
4. mAP 是多个类别 AP 的平均。
5. `mAP50` 更关注是否大致找到目标。
6. `mAP75` 更关注框的位置是否准确。
7. `mAP50-95` 是更严格、更综合的检测质量评价。
8. 高 confidence FP 会严重损害 AP，因为它会排在 PR 统计序列前面。

## 12. 下一步

Day7 建议继续做：

```text
解读 COCO128 验证结果，并形成第一份正式实验报告
```

重点包括：

- 读取 `mAP50`、`mAP75`、`mAP50-95`。
- 用 Day6 的概念解释这些指标。
- 写出一段规范的实验分析。
- 为后续模型规模比较、`imgsz` 比较、NMS IoU 参数实验做准备。

mAP50 只要求预测框和真实框的 IoU 达到 0.5，因此评价标准相对宽松；
mAP50-95 会综合 IoU=0.50 到 0.95 的多个阈值，其中高 IoU 阈值对定位精度要求更高。
因此 mAP50 明显高于 mAP50-95，说明模型能够大致找到目标，但在更严格定位条件下表现下降，预测框的位置精度仍有提升空间。


