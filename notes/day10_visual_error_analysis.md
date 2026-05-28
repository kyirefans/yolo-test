# Day10：MOT17 可视化误差分析

## 1. 实验目的

Day9 已经通过 MOT17 的 GT 标注，验证了 `confidence threshold` 和 `NMS IoU threshold` 对 TP / FP / FN、Precision、Recall 的影响。

Day10 的目标是进一步观察：

```text
这些 FP、FN、重复框在图像上到底长什么样。
```

也就是说，Day10 要从纯数字分析进入可视化误差分析。

## 2. 可视化脚本

新增脚本：

```text
scripts/visualize_mot_detections.py
```

它会读取：

```text
1. MOT17 图像帧
2. MOT17 gt/gt.txt
3. YOLO detections.json
```

然后将预测框和 GT 画在同一张图上。

可视化颜色约定：

| 类型 | 颜色 | 含义 |
| --- | --- | --- |
| GT | 绿色 | 原始真实框 |
| TP | 蓝色 | 成功匹配 GT 的预测框 |
| FP | 红色 | 没有匹配到 GT 的错误预测框 |
| FN | 黄色 | 没有被预测框匹配到的 GT |

每张图左上角会显示：

```text
Frame ID
TP / FP / FN
Precision / Recall
颜色说明
```

## 3. 可视化输出位置

本次选择了 3 个典型帧：

```text
frame_000016.jpg
frame_000345.jpg
frame_000371.jpg
```

并对比 3 组参数：

```text
conf=0.25, nms_iou=0.70
conf=0.50, nms_iou=0.70
conf=0.25, nms_iou=0.90
```

输出目录：

```text
experiments/day10_visual_error_analysis/conf025_iou070/
experiments/day10_visual_error_analysis/conf050_iou070/
experiments/day10_visual_error_analysis/conf025_iou090/
```

## 4. 三组参数的帧级结果

| setting | frame | TP | FP | FN | predictions | GT | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conf025_iou070 | 16 | 7 | 8 | 16 | 15 | 23 | 0.467 | 0.304 |
| conf025_iou070 | 345 | 4 | 5 | 29 | 9 | 33 | 0.444 | 0.121 |
| conf025_iou070 | 371 | 5 | 2 | 31 | 7 | 36 | 0.714 | 0.139 |
| conf050_iou070 | 16 | 4 | 0 | 19 | 4 | 23 | 1.000 | 0.174 |
| conf050_iou070 | 345 | 4 | 0 | 29 | 4 | 33 | 1.000 | 0.121 |
| conf050_iou070 | 371 | 5 | 0 | 31 | 5 | 36 | 1.000 | 0.139 |
| conf025_iou090 | 16 | 9 | 20 | 14 | 29 | 23 | 0.310 | 0.391 |
| conf025_iou090 | 345 | 4 | 7 | 29 | 11 | 33 | 0.364 | 0.121 |
| conf025_iou090 | 371 | 5 | 3 | 31 | 8 | 36 | 0.625 | 0.139 |

## 5. Frame 16 分析

默认参数：

```text
conf=0.25, nms_iou=0.70
TP = 7
FP = 8
FN = 16
Precision = 0.467
Recall = 0.304
```

这说明该帧中模型检测到了一部分行人，但误检和漏检都比较明显。

当提高 confidence threshold：

```text
conf=0.50, nms_iou=0.70
TP = 4
FP = 0
FN = 19
Precision = 1.000
Recall = 0.174
```

可以看到：

```text
FP 被完全过滤掉，Precision 上升到 1.0；
但 TP 从 7 降到 4，FN 从 16 增加到 19，Recall 下降。
```

这说明提高 `conf` 会让模型更保守，误检减少，但漏检增加。

当提高 NMS IoU：

```text
conf=0.25, nms_iou=0.90
TP = 9
FP = 20
FN = 14
Precision = 0.310
Recall = 0.391
```

可以看到：

```text
TP 从 7 增加到 9，Recall 从 0.304 提高到 0.391；
但 FP 从 8 增加到 20，Precision 从 0.467 降到 0.310。
```

这说明更宽松的 NMS 保留了更多框，其中一部分匹配到 GT，但更多变成了 FP。

## 6. Frame 345 分析

默认参数：

```text
TP = 4
FP = 5
FN = 29
Precision = 0.444
Recall = 0.121
```

该帧 GT 数量为 33，但模型只匹配到 4 个目标，说明漏检非常严重。

提高 confidence threshold 后：

```text
TP = 4
FP = 0
FN = 29
Precision = 1.000
Recall = 0.121
```

这说明该帧中提高 `conf` 主要删除了 FP，没有进一步减少 TP，因此 Precision 变好，但 Recall 没有改善。

提高 NMS IoU 后：

```text
TP = 4
FP = 7
FN = 29
Precision = 0.364
Recall = 0.121
```

这说明更宽松的 NMS 没有帮助模型找回更多 GT，只是额外保留了更多 FP。

## 7. Frame 371 分析

默认参数：

```text
TP = 5
FP = 2
FN = 31
Precision = 0.714
Recall = 0.139
```

该帧 GT 数量为 36，但只匹配到 5 个目标。Precision 不算特别低，但 Recall 很低。

这类情况说明：

```text
模型预测出来的框大多比较准；
但大量真实目标根本没有被预测出来。
```

提高 confidence threshold 后：

```text
TP = 5
FP = 0
FN = 31
Precision = 1.000
Recall = 0.139
```

这里提高 `conf` 删除了少量 FP，但没有改变 TP / FN，因此 Recall 不变。

提高 NMS IoU 后：

```text
TP = 5
FP = 3
FN = 31
Precision = 0.625
Recall = 0.139
```

更宽松的 NMS 只增加了 FP，没有带来额外 TP。

## 8. Day10 观察结论

### 8.1 提高 conf 的效果

从 3 个典型帧可以看到：

```text
conf 从 0.25 提高到 0.50 后，FP 明显减少。
```

但代价是：

```text
部分 TP 被过滤掉，FN 可能增加，Recall 可能下降。
```

因此：

```text
提高 conf 适合减少误检，但不适合解决漏检。
```

### 8.2 提高 NMS IoU 的效果

`nms_iou=0.90` 会让 NMS 更宽松。

观察结果：

```text
某些帧 TP 略有增加；
但 FP 增加更明显；
Precision 明显下降。
```

因此：

```text
过高的 NMS IoU 容易保留重复框或冗余框。
```

### 8.3 MOT17 上的主要问题

MOT17 是密集行人场景，存在：

```text
1. 小目标；
2. 遮挡；
3. 密集人群；
4. 远距离行人；
5. 部分目标可见区域很小。
```

当前 YOLO11n 的主要问题不是单纯误检，而是：

```text
Recall 偏低，漏检较多。
```

仅靠提高 `conf` 无法解决漏检，反而会让 Recall 更低。

## 9. Day10 总结

Day10 的核心收获是：

```text
数字指标必须结合图像可视化理解。
```

同样是 Precision / Recall 变化，可视化后能看到：

```text
1. 红框 FP 是什么类型的误检；
2. 黄色 FN 是否集中在小目标、遮挡目标、密集区域；
3. NMS IoU 过高时是否出现重复框；
4. conf 提高后哪些预测框被删除。
```

当前结论：

> 在 MOT17 场景中，提高 confidence threshold 可以明显减少 FP，但会牺牲 Recall；提高 NMS IoU 会保留更多框，可能略微提高 Recall，但容易显著增加 FP。YOLO11n 在密集行人场景中的主要瓶颈是漏检，后续应重点关注小目标、遮挡和密集目标检测问题。

## 10. 下一步

Day11 可以进入：

```text
模型规模对比：yolo11n vs yolo11s
```

目标是观察更大模型是否能改善：

```text
1. MOT17 上的 Recall；
2. 小目标和遮挡目标漏检；
3. mAP50 / mAP75 / mAP50-95；
4. 推理速度和模型大小。
```

建议继续使用同一套参数和同一套 MOT17 序列，这样对比才公平。
