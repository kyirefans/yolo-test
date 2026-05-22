# Day9：Confidence Threshold 与 NMS IoU 实验报告

## 1. 实验目的

Day9 的目标是理解两个推理阶段参数对 YOLO 检测输出的影响：

```text
confidence threshold
NMS IoU threshold
```

本次实验重点不是直接计算 Precision、Recall 或 mAP，而是先观察不同参数如何影响：

```text
1. 总预测框数量
2. 检测类别数量
3. 平均置信度
4. 主要类别分布
5. 模型输出是否更保守或更宽松
```

需要明确：

```text
confidence threshold 控制低置信度预测框是否保留。
NMS IoU threshold 控制高度重叠预测框是否被认为重复。
```

## 2. 实验设置

检测脚本：

```text
src/detect.py
```

统计脚本：

```text
scripts/analyze_detections.py
```

输入视频：

```text
data/videos/test.mp4
```

模型：

```text
yolo11n.pt
```

本次实验前，对 `src/detect.py` 增加了 `--iou` 参数，使脚本支持配置 NMS IoU threshold。

核心参数传递逻辑：

```python
results = model.predict(
    source=source,
    conf=conf,
    iou=iou,
    save=True,
    project=output_dir,
    name=name,
    exist_ok=True,
)
```

## 3. 实验分组

本次共运行 5 组实验：

| 实验 | conf | nms_iou | 输出目录 |
| --- | ---: | ---: | --- |
| A | 0.25 | 0.70 | `day9_conf025_iou070` |
| B | 0.50 | 0.70 | `day9_conf050_iou070` |
| C | 0.75 | 0.70 | `day9_conf075_iou070` |
| D | 0.25 | 0.50 | `day9_conf025_iou050` |
| E | 0.25 | 0.90 | `day9_conf025_iou090` |

实验结果保存在：

```text
experiments/day9_conf025_iou050/
experiments/day9_conf025_iou070/
experiments/day9_conf025_iou090/
experiments/day9_conf050_iou070/
experiments/day9_conf075_iou070/
```

## 4. 整体结果

| 实验 | conf | nms_iou | total_detections | class_count | avg_confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.25 | 0.70 | 3002 | 19 | 0.6185 |
| B | 0.50 | 0.70 | 1944 | 11 | 0.7616 |
| C | 0.75 | 0.70 | 1171 | 1 | 0.8562 |
| D | 0.25 | 0.50 | 2783 | 19 | 0.6387 |
| E | 0.25 | 0.90 | 4276 | 19 | 0.5469 |

其中 `avg_confidence` 是按类别统计结果加权计算得到的整体平均置信度。

## 5. Confidence Threshold 对输出的影响

为了单独观察 confidence threshold 的影响，固定：

```text
nms_iou = 0.70
```

对比三组实验：

| conf | total_detections | class_count | avg_confidence | 主要类别 |
| ---: | ---: | ---: | ---: | --- |
| 0.25 | 3002 | 19 | 0.6185 | `person:2655`, `tv:102`, `chair:68` |
| 0.50 | 1944 | 11 | 0.7616 | `person:1898`, `sports ball:10`, `baseball glove:8` |
| 0.75 | 1171 | 1 | 0.8562 | `person:1171` |

可以看到：

```text
conf 从 0.25 提高到 0.50：
预测框数量从 3002 降到 1944。

conf 从 0.50 提高到 0.75：
预测框数量从 1944 降到 1171。

conf = 0.75 时：
检测类别只剩 person 一类。
```

结论：

```text
confidence threshold 越高，保留的预测框越少，类别数量越少，平均置信度越高。
```

解释：

```text
提高 confidence threshold 会过滤掉更多低置信度预测框。
因此模型输出更保守，预测框数量下降，平均置信度上升。
```

与 Precision / Recall 的关系：

```text
FP 可能减少 -> Precision 可能上升。
低置信度 TP 也可能被过滤 -> FN 可能增加，Recall 可能下降。
```

所以，提高 `conf` 并不代表模型本身变强，而是推理输出策略变得更保守。

## 6. NMS IoU Threshold 对输出的影响

为了单独观察 NMS IoU threshold 的影响，固定：

```text
conf = 0.25
```

对比三组实验：

| nms_iou | total_detections | class_count | avg_confidence | 主要类别 |
| ---: | ---: | ---: | ---: | --- |
| 0.50 | 2783 | 19 | 0.6387 | `person:2445`, `tv:102`, `chair:68` |
| 0.70 | 3002 | 19 | 0.6185 | `person:2655`, `tv:102`, `chair:68` |
| 0.90 | 4276 | 19 | 0.5469 | `person:3874`, `chair:105`, `tv:104` |

可以看到：

```text
nms_iou 从 0.50 提高到 0.70：
预测框数量从 2783 增加到 3002。

nms_iou 从 0.70 提高到 0.90：
预测框数量从 3002 增加到 4276。
```

结论：

```text
NMS IoU threshold 越高，NMS 越宽松，保留的重叠框越多。
NMS IoU threshold 越低，NMS 越严格，删除的重叠框越多。
```

解释：

```text
NMS 会比较预测框之间的 IoU。
当两个预测框重叠程度超过 NMS IoU threshold 时，低置信度框可能被删除。
```

因此：

```text
nms_iou = 0.50:
两个框只要重叠超过 0.50，就更容易被认为重复，删除更多框。

nms_iou = 0.90:
两个框必须重叠超过 0.90，才更容易被认为重复，删除更少框。
```

这也是为什么：

```text
conf=0.25, iou=0.90 -> 4276 个框
conf=0.25, iou=0.70 -> 3002 个框
```

`iou=0.90` 的 NMS 更宽松，很多在 `iou=0.70` 下可能被压掉的重叠框被保留下来了。

## 7. Conf 与 NMS IoU 的区别

两个参数都会影响最终预测框数量，但控制机制不同。

| 参数 | 比较对象 | 控制内容 | 典型影响 |
| --- | --- | --- | --- |
| `confidence threshold` | 预测框自己 | 置信度是否足够高 | 过滤低置信度框 |
| `NMS IoU threshold` | 预测框 vs 预测框 | 两个框是否被认为重复 | 删除重叠框 |

简单记忆：

```text
conf 管“这个框自己够不够可信”。
NMS IoU 管“这些框之间是不是太重复”。
```

## 8. 与 Day8 错误分析的关系

Day8 中观察到部分图像存在大量 FP，例如：

```text
000000000042.jpg:
TP = 1
FP = 225
FN = 0
Precision = 0.0044
Recall = 1.0000
```

Day9 的参数实验说明，面对 FP 较多的问题，可以考虑两个方向：

```text
1. 提高 confidence threshold，过滤低置信度误检框。
2. 降低 NMS IoU threshold，压掉更多高度重叠的重复框。
```

但这两个方向都有风险：

```text
conf 过高:
可能删除一部分真实目标对应的低置信度 TP，导致 Recall 下降。

NMS IoU 过低:
可能把相邻目标或合理预测框误认为重复，导致漏检增加。
```

所以参数调整需要结合 GT 验证，而不能只看输出框数量。

## 9. 本次实验的限制

本次实验基于视频推理输出和 `detections.json` 统计，没有逐帧 GT 标注。

因此，本次实验能够严格分析：

```text
1. 输出框数量变化
2. 类别数量变化
3. 平均置信度变化
4. 参数对输出保守程度的影响
```

但不能严格计算：

```text
1. Precision
2. Recall
3. FP / FN
4. AP / mAP
```

因为这些指标都需要 GT 参与匹配。

因此，当前关于 Precision / Recall 的判断只能是合理推测：

```text
提高 conf 可能减少 FP，但也可能增加 FN。
降低 NMS IoU 可能减少重复 FP，但也可能误删 TP。
```

## 10. Day9 结论

本次实验验证了两个关键规律：

```text
1. confidence threshold 越高，输出框越少，平均置信度越高，模型越保守。
2. NMS IoU threshold 越高，NMS 越宽松，保留的重叠框越多。
```

当前实验结果可以总结为：

```text
conf 从 0.25 提高到 0.75:
total_detections 从 3002 降到 1171。
class_count 从 19 降到 1。
avg_confidence 从 0.6185 升到 0.8562。

nms_iou 从 0.50 提高到 0.90:
total_detections 从 2783 升到 4276。
class_count 保持 19。
avg_confidence 从 0.6387 降到 0.5469。
```

最终结论：

> `confidence threshold` 主要控制低置信度框是否保留，`NMS IoU threshold` 主要控制高度重叠框是否被删除。二者都会影响预测框数量，但机制不同。后续如果要判断这些参数是否真正提高模型性能，需要在带 GT 的验证集上比较 Precision、Recall 和 mAP。

## 11. 下一步

Day10 建议进入：

```text
可视化检测结果，人工观察 FP / FN 和重复框
```

重点任务：

```text
1. 对比 conf=0.25、0.50、0.75 的可视化结果。
2. 对比 nms_iou=0.50、0.70、0.90 的可视化结果。
3. 观察哪些框是明显误检。
4. 观察哪些框是重复框。
5. 选择典型帧写案例分析。
```

更进一步，可以在 COCO128 验证集上设计参数实验：

```text
不同 conf / iou 设置下重新验证，比较 mAP50、mAP75、mAP50-95。
```

这样才能从“输出数量分析”进入“真实性能评估”。
