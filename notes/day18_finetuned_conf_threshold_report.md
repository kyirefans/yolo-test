# Day18: 微调模型 Confidence Threshold 实验报告

## 1. 实验目标

Day17 已经完成了预训练 YOLO11s 和 MOT17 微调 YOLO11s 的对比。

Day17 的核心结论是：

```text
微调后 Recall 明显提升，说明模型更适应 MOT17 行人场景；
但 Precision 下降，说明模型也输出了更多 false positives。
```

所以 Day18 的目标不是继续训练模型，而是研究：

```text
同一个微调模型，在不同 confidence threshold 下，Precision / Recall 会如何变化？
```

这个实验要回答一个实际问题：

```text
微调后的模型应该用哪个 confidence threshold 作为后续 tracking 的输入？
```

## 2. 实验设置

模型：

```text
runs/detect/experiments/day16_train_mot17_yolo_small/yolo11s_imgsz960_epochs5/weights/best.pt
```

数据：

```text
datasets/MOT17-02-SDP
```

评估脚本：

```text
scripts/evaluate_mot_detections.py
```

评估对象：

```text
class_name = person
evaluation IoU threshold = 0.5
marked_only = true
```

实验变量：

```text
confidence threshold = 0.25 / 0.35 / 0.50 / 0.65
```

## 3. 实验结果

| Confidence | Predictions | TP | FP | FN | Precision | Recall | F1 | Avg IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 19375 | 12877 | 6498 | 5704 | 0.6646 | 0.6930 | 0.6785 | 0.7992 |
| 0.35 | 14811 | 11744 | 3067 | 6837 | 0.7929 | 0.6320 | 0.7044 | 0.8114 |
| 0.50 | 11043 | 10236 | 807 | 8345 | 0.9269 | 0.5509 | 0.6912 | 0.8284 |
| 0.65 | 8569 | 8442 | 127 | 10139 | 0.9852 | 0.4543 | 0.6220 | 0.8487 |

F1 的计算方式：

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

F1 用来衡量 Precision 和 Recall 的综合平衡。

## 4. 结果解释

### 4.1 confidence 越高，预测框越少

从结果看：

```text
conf=0.25: 19375 个预测框
conf=0.35: 14811 个预测框
conf=0.50: 11043 个预测框
conf=0.65: 8569 个预测框
```

这是因为 confidence threshold 越高，低置信度预测越容易被过滤。

所以模型输出会变得更保守：

```text
只保留更有把握的检测框。
```

### 4.2 Precision 随阈值升高而上升

Precision 的变化：

```text
0.6646 -> 0.7929 -> 0.9269 -> 0.9852
```

含义是：

```text
阈值越高，留下来的框越可靠，误检越少。
```

例如 `conf=0.65` 时：

```text
FP = 127
Precision = 0.9852
```

这说明模型几乎只保留非常确定的行人框。

### 4.3 Recall 随阈值升高而下降

Recall 的变化：

```text
0.6930 -> 0.6320 -> 0.5509 -> 0.4543
```

含义是：

```text
阈值越高，越多真实行人会因为置信度不够高被过滤掉。
```

例如 `conf=0.65` 时：

```text
FN = 10139
Recall = 0.4543
```

这说明虽然留下来的框很准，但漏掉了大量行人。

### 4.4 Avg IoU 随阈值升高而上升

Avg IoU 的变化：

```text
0.7992 -> 0.8114 -> 0.8284 -> 0.8487
```

这不是说高阈值让模型突然学会了更精确定位，而是因为：

```text
低置信度、定位较差、遮挡严重、小目标的框被过滤掉了；
剩下的通常是更容易检测、定位更稳定的目标。
```

所以 Avg IoU 上升代表：

```text
被保留下来的匹配框质量更高。
```

但它同时付出了 Recall 下降的代价。

## 5. 为什么 conf=0.35 当前最合适

从 F1 看：

```text
conf=0.35 的 F1 = 0.7044
```

是四组实验中最高的。

这说明 `conf=0.35` 在当前实验中取得了最好的 Precision / Recall 平衡：

```text
比 conf=0.25 少很多 FP；
比 conf=0.50 和 conf=0.65 保留更多真实目标。
```

具体对比：

```text
conf=0.25 -> FP 太多，可能给 tracking 带来假轨迹。
conf=0.35 -> Precision 和 Recall 比较平衡。
conf=0.50 -> Precision 高，但 Recall 已经明显下降。
conf=0.65 -> 几乎不误检，但漏检太多。
```

所以当前推荐：

```text
后续 tracking 默认使用 conf=0.35 作为检测输入。
```

## 6. 不同阈值适合的场景

### 6.1 需要尽量找全目标

适合：

```text
conf=0.25
```

场景：

```text
宁愿多报一些，也不希望漏掉人。
```

缺点：

```text
FP 多，后续需要更强的过滤或跟踪算法处理。
```

### 6.2 需要检测和跟踪平衡

适合：

```text
conf=0.35
```

场景：

```text
既不能漏掉太多行人，也不能产生太多假框。
```

这是当前项目最推荐的默认阈值。

### 6.3 需要高置信度检测结果

适合：

```text
conf=0.50 或 conf=0.65
```

场景：

```text
只想保留非常可靠的检测结果，误报代价很高。
```

缺点：

```text
Recall 明显下降，容易漏掉远处、小目标、遮挡目标。
```

## 7. Day18 结论

Day18 的核心结论：

```text
微调提升的是模型对 MOT17 行人场景的适应能力；
confidence threshold 决定的是如何使用这个微调后的模型输出。
```

当前最重要的判断是：

```text
没有唯一的最好阈值，只有最适合任务目标的阈值。
```

对于当前项目，下一步要进入多目标跟踪，所以不能只追求单帧 Precision。

推荐选择：

```text
conf=0.35
```

原因：

```text
它取得最高 F1，能够在误检和漏检之间保持相对平衡。
```

