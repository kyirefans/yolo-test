# Day11：YOLO11n vs YOLO11s 模型规模对比

## 1. 实验目的

Day10 的可视化分析说明，YOLO11n 在 MOT17 密集行人场景中的主要问题是：

```text
Recall 偏低，漏检较多。
```

Day11 的目标是验证：

```text
换成更大的 YOLO11s 后，是否能改善 MOT17 上的检测效果？
```

重点关注：

```text
1. TP 是否增加；
2. FN 是否减少；
3. Recall 是否提高；
4. FP 是否明显增加；
5. Precision 是否还能保持；
6. 匹配框的平均 IoU 是否提高。
```

## 2. 实验设置

数据集：

```text
MOT17-02-SDP
```

图像帧：

```text
datasets/MOT17-02-SDP/img1
```

GT：

```text
datasets/MOT17-02-SDP/gt/gt.txt
```

评估对象：

```text
YOLO person detections vs MOT17 pedestrian GT
```

固定参数：

```text
conf = 0.25
nms_iou = 0.70
eval_iou = 0.50
```

对比模型：

```text
yolo11n.pt
yolo11s.pt
```

## 3. 整体结果对比

| model | predictions | GT | TP | FP | FN | Precision | Recall | Avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n | 5732 | 18581 | 4591 | 1141 | 13990 | 0.8009 | 0.2471 | 0.8098 |
| yolo11s | 6530 | 18581 | 5176 | 1354 | 13405 | 0.7926 | 0.2786 | 0.8213 |

## 4. 整体结果解释

和 YOLO11n 相比，YOLO11s：

```text
predictions: 5732 -> 6530，增加 798
TP:          4591 -> 5176，增加 585
FP:          1141 -> 1354，增加 213
FN:         13990 -> 13405，减少 585
Recall:   0.2471 -> 0.2786，提高 0.0315
Precision:0.8009 -> 0.7926，略微下降 0.0083
Avg IoU:  0.8098 -> 0.8213，略有提高
```

结论：

```text
YOLO11s 确实找回了更多行人目标，使 Recall 上升；
同时 FP 也有所增加，使 Precision 略微下降；
但 Precision 下降幅度很小，整体上 YOLO11s 比 YOLO11n 更适合该 MOT17 序列。
```

## 5. 典型帧对比

选择 Day10 中的 3 个典型帧：

```text
frame 16
frame 345
frame 371
```

帧级结果：

| model | frame | TP | FP | FN | predictions | GT | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n | 16 | 7 | 8 | 16 | 15 | 23 | 0.467 | 0.304 |
| yolo11n | 345 | 4 | 5 | 29 | 9 | 33 | 0.444 | 0.121 |
| yolo11n | 371 | 5 | 2 | 31 | 7 | 36 | 0.714 | 0.139 |
| yolo11s | 16 | 7 | 1 | 16 | 8 | 23 | 0.875 | 0.304 |
| yolo11s | 345 | 6 | 3 | 27 | 9 | 33 | 0.667 | 0.182 |
| yolo11s | 371 | 7 | 6 | 29 | 13 | 36 | 0.538 | 0.194 |

## 6. 典型帧分析

### 6.1 Frame 16

YOLO11n：

```text
TP = 7
FP = 8
FN = 16
Precision = 0.467
Recall = 0.304
```

YOLO11s：

```text
TP = 7
FP = 1
FN = 16
Precision = 0.875
Recall = 0.304
```

分析：

```text
YOLO11s 没有增加 TP，但显著减少 FP。
因此 Recall 不变，Precision 明显提高。
```

### 6.2 Frame 345

YOLO11n：

```text
TP = 4
FP = 5
FN = 29
Precision = 0.444
Recall = 0.121
```

YOLO11s：

```text
TP = 6
FP = 3
FN = 27
Precision = 0.667
Recall = 0.182
```

分析：

```text
YOLO11s 多找回 2 个目标，同时 FP 从 5 降到 3。
该帧上 YOLO11s 同时改善了 Precision 和 Recall。
```

### 6.3 Frame 371

YOLO11n：

```text
TP = 5
FP = 2
FN = 31
Precision = 0.714
Recall = 0.139
```

YOLO11s：

```text
TP = 7
FP = 6
FN = 29
Precision = 0.538
Recall = 0.194
```

分析：

```text
YOLO11s 多找回 2 个目标，使 Recall 提高；
但 FP 从 2 增加到 6，Precision 下降。
```

这说明更大的模型有时会检测更多目标，但也可能带来更多误检。

## 7. 可视化输出

YOLO11s 的典型帧可视化输出在：

```text
experiments/day11_visual_model_compare/yolo11s_conf025_iou070/
```

可以和 Day10 的 YOLO11n 输出对比：

```text
experiments/day10_visual_error_analysis/conf025_iou070/
```

重点观察：

```text
1. yolo11s 是否减少黄色 FN 框；
2. yolo11s 是否增加红色 FP 框；
3. yolo11s 的蓝色 TP 框是否覆盖更多遮挡或小目标；
4. 同一帧中，yolo11s 是否比 yolo11n 更稳定。
```

## 8. Day11 结论

本次实验说明：

```text
YOLO11s 相比 YOLO11n 能提升 MOT17 上的 Recall。
```

但提升不是免费的：

```text
YOLO11s 输出更多预测框，TP 增加，同时 FP 也增加。
Precision 略微下降，但下降幅度较小。
```

综合来看：

> 在 MOT17-02-SDP 密集行人场景中，YOLO11s 比 YOLO11n 更适合当前任务，因为它提高了 Recall，并且 Precision 仍保持在接近 0.79 的水平。

不过，YOLO11s 仍然没有彻底解决漏检问题：

```text
Recall = 0.2786
```

这说明大量 GT 仍未被检测到。后续需要继续考虑：

```text
1. 更大模型；
2. 更高输入分辨率 imgsz；
3. 针对 MOT17 行人场景微调；
4. 小目标和遮挡场景的专门优化。
```

## 9. 下一步

Day12 建议做：

```text
输入尺寸 imgsz 对检测效果的影响
```

固定模型和参数：

```text
model = yolo11s.pt
conf = 0.25
nms_iou = 0.70
```

对比：

```text
imgsz = 640
imgsz = 960
imgsz = 1280
```

重点观察：

```text
1. Recall 是否提高；
2. 小目标和远距离行人是否更容易被检测；
3. FP 是否增加；
4. 推理速度是否明显下降。
```
