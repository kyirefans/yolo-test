# Day8：COCO128 验证实验报告

## 1. 实验目的

本次实验的目标不是继续学习新的指标公式，而是把 Day6 和 Day7 学到的评价概念用于解释真实验证结果。

重点问题包括：

```text
1. YOLO11n 在 COCO128 上整体检测能力如何？
2. mAP50、mAP75、mAP50-95 的差距说明什么？
3. 单张图像上的 TP / FP / FN 如何解释？
4. 当前模型主要问题是误检、漏检，还是定位精度不足？
5. 下一步应该设计什么实验继续分析？
```

## 2. 实验设置

验证脚本：

```text
scripts/validate_coco128.py
```

模型：

```text
yolo11n.pt
```

数据集：

```text
COCO128
```

输入尺寸：

```text
imgsz = 640
```

验证代码核心部分：

```python
model = YOLO("yolo11n.pt")

results = model.val(
    data="coco128.yaml",
    imgsz=640,
    plots=True,
    verbose=True,
)
```

实验结果保存位置：

```text
experiments/day5_coco128_validation/coco128_validation_summary.json
```

## 3. 整体指标

本次 COCO128 验证结果为：

```text
mAP50    = 0.6699
mAP75    = 0.5390
mAP50-95 = 0.5026
```

对应含义：

```text
mAP50:
在 IoU=0.5 的宽松匹配条件下，多类别 AP 的平均值。

mAP75:
在 IoU=0.75 的更严格匹配条件下，多类别 AP 的平均值。

mAP50-95:
在 IoU=0.50 到 0.95 多个阈值下计算 AP 后再平均，是更严格、更综合的检测指标。
```

## 4. 整体指标解释

`mAP50 = 0.6699` 说明模型在较宽松的 IoU=0.5 条件下具备一定目标检测能力，能够大致找到不少目标。

`mAP75 = 0.5390` 低于 `mAP50`，说明当定位要求提高后，一部分预测框无法满足更严格的 IoU 匹配标准。

`mAP50-95 = 0.5026` 进一步说明，模型在多个 IoU 阈值下的综合定位质量仍有提升空间。

规范实验结论：

> 本次使用 YOLO11n 在 COCO128 数据集上进行验证。实验结果显示，模型的 mAP50 为 0.6699，mAP75 为 0.5390，mAP50-95 为 0.5026。mAP50 高于 mAP75 和 mAP50-95，说明模型在较宽松的 IoU=0.5 条件下具备一定目标检测能力；但当 IoU 阈值提高后，性能明显下降，表明模型的定位精度仍有提升空间。

## 5. 单图像案例分析

### 5.1 检测效果较好的图像

图像：

```text
000000000626.jpg
```

验证结果：

```text
GT = 2
TP = 2
FP = 0
FN = 0
Precision = 1.0000
Recall = 1.0000
```

计算：

```text
Precision = TP / (TP + FP)
          = 2 / (2 + 0)
          = 1.0000

Recall = TP / (TP + FN)
       = 2 / (2 + 0)
       = 1.0000
```

分析：

> 图像 `000000000626.jpg` 中共有 2 个真实目标，模型成功检测出全部目标且没有产生误检，因此 Precision 和 Recall 均为 1.0，说明该图像上的检测结果较理想。

### 5.2 Recall 高但 Precision 极低的图像

图像：

```text
000000000042.jpg
```

验证结果：

```text
GT = 1
TP = 1
FP = 225
FN = 0
Precision = 0.0044
Recall = 1.0000
```

计算：

```text
Precision = TP / (TP + FP)
          = 1 / (1 + 225)
          = 1 / 226
          = 0.0044

Recall = TP / (TP + FN)
       = 1 / (1 + 0)
       = 1.0000
```

分析：

> 图像 `000000000042.jpg` 中只有 1 个真实目标，模型成功检测到了该目标，因此 Recall 为 1.0。但模型同时产生了 225 个 FP，导致 Precision 仅为 0.0044。这说明该图像虽然没有漏检真实目标，但误检数量极多，整体检测质量较差。

这个案例说明：

```text
高 Recall 不一定代表模型表现好。
如果 FP 很多，Precision 会非常低。
```

### 5.3 Precision 和 Recall 都较低的图像

图像：

```text
000000000540.jpg
```

验证结果：

```text
GT = 20
TP = 6
FP = 192
FN = 14
Precision = 0.0303
Recall = 0.3000
```

计算：

```text
Precision = TP / (TP + FP)
          = 6 / (6 + 192)
          = 6 / 198
          = 0.0303

Recall = TP / (TP + FN)
       = 6 / (6 + 14)
       = 6 / 20
       = 0.3000
```

分析：

> 图像 `000000000540.jpg` 中共有 20 个真实目标，但模型只匹配到 6 个，漏检 14 个，同时产生 192 个 FP，导致 Precision 和 Recall 都较低。这说明模型在该图像上既存在明显漏检，也存在大量误检。

### 5.4 没有 GT 但产生预测框的图像

图像：

```text
000000000508.jpg
```

验证结果：

```text
GT = 0
TP = 0
FP = 31
FN = 0
Precision = 0.0000
Recall = 0.0000
```

分析：

> 图像 `000000000508.jpg` 在当前标注中没有 GT 目标，但模型仍然输出了 31 个预测框。这些预测框无法匹配任何真实目标，因此全部属于 FP。该图像是分析纯误检问题的典型案例。

## 6. 当前模型问题总结

根据整体指标和单图像案例，可以总结出三个主要问题。

### 6.1 定位精度仍有提升空间

证据：

```text
mAP50    = 0.6699
mAP75    = 0.5390
mAP50-95 = 0.5026
```

解释：

```text
mAP50 高于 mAP75 和 mAP50-95。
这说明模型在宽松 IoU 条件下表现更好，
但当 IoU 阈值提高后，部分预测框不能满足严格定位要求。
```

### 6.2 部分图像误检非常多

证据：

```text
000000000042.jpg:
TP = 1
FP = 225
FN = 0
Precision = 0.0044
Recall = 1.0000
```

解释：

```text
模型找到了真实目标，但同时产生大量 FP。
这类问题会显著降低 Precision。
```

### 6.3 部分图像同时存在误检和漏检

证据：

```text
000000000540.jpg:
GT = 20
TP = 6
FP = 192
FN = 14
Precision = 0.0303
Recall = 0.3000
```

解释：

```text
模型只找回少量真实目标，同时产生大量错误预测。
这类图像说明模型在复杂场景下仍然不稳定。
```

## 7. 与 confidence threshold 的关系

之前的视频推理实验显示：

```text
conf = 0.25 -> detections = 3002, avg_conf = 0.6185
conf = 0.50 -> detections = 1944, avg_conf = 0.7616
conf = 0.75 -> detections = 1171, avg_conf = 0.8562
```

结论：

```text
confidence threshold 越高，保留的预测框越少，平均置信度越高，模型输出越保守。
```

如果某些图像 FP 很多，提高 `confidence threshold` 可能会减少低置信度误检框，使 Precision 上升。

但同时也要注意：

```text
提高 confidence threshold 可能删除一部分原本正确的低置信度 TP，
从而导致 FN 增加、Recall 下降。
```

所以 confidence threshold 的调整本质上是在控制：

```text
Precision 与 Recall 的权衡
```

## 8. 与 NMS IoU threshold 的关系

`confidence threshold` 控制的是：

```text
预测框自身置信度是否足够高
```

`NMS IoU threshold` 控制的是：

```text
预测框之间是否被认为重复
```

如果某张图中存在大量重复框，那么问题可能和 NMS 有关。

一般来说：

```text
NMS IoU threshold 较低:
更容易认为两个框重复，删除更多框，输出更少。

NMS IoU threshold 较高:
更宽松，允许更多重叠框保留，输出可能更多。
```

因此，后续应同时关注 `confidence threshold` 和 `NMS IoU threshold` 对 FP / FN 的影响。

## 9. Day9 实验计划

Day9 建议研究：

```text
confidence threshold 与 NMS IoU threshold 对检测结果的影响
```

建议实验表：

| 实验 | conf | nms_iou | 预期 |
| --- | ---: | ---: | --- |
| A | 0.25 | 0.70 | 默认设置，预测较多 |
| B | 0.50 | 0.70 | 过滤更多低置信度框 |
| C | 0.75 | 0.70 | 输出更保守，可能漏检 |
| D | 0.25 | 0.50 | NMS 更严格，重复框更少 |
| E | 0.25 | 0.90 | NMS 更宽松，重复框可能更多 |

Day9 需要观察：

```text
1. 总预测框数量如何变化？
2. 类别数量如何变化？
3. FP 是否减少？
4. FN 是否增加？
5. Precision 是否上升？
6. Recall 是否下降？
7. mAP 是否变化？
```

## 10. Day8 总结

本次 Day8 的核心收获是：

```text
评价模型不能只看一个指标。
```

`mAP50` 可以说明模型在宽松条件下的检测能力，`mAP75` 和 `mAP50-95` 更能反映定位精度。

单图像分析中：

```text
高 Recall + 低 Precision:
说明模型找到了目标，但误检很多。

低 Recall + 低 Precision:
说明模型既漏检，又误检。

GT=0 但 FP>0:
说明模型在没有目标的图像上产生了误检。
```

当前 YOLO11n 在 COCO128 上具备一定检测能力，但仍存在：

```text
1. 定位精度不足；
2. 部分图像误检较多；
3. 部分复杂图像漏检和误检同时存在。
```

后续应通过调整 `confidence threshold`、`NMS IoU threshold`、模型规模和输入尺寸，继续分析 FP / FN 与 mAP 的变化。
