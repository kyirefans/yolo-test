# Day12：输入尺寸 imgsz 对检测效果的影响

## 1. 实验目的

Day11 发现 YOLO11s 相比 YOLO11n 能提高 MOT17 上的 Recall，但漏检仍然明显。

Day12 继续研究：

```text
增大输入尺寸 imgsz，是否能改善小目标、远距离目标和密集行人的检测？
```

核心问题：

```text
1. imgsz 增大后，TP 是否增加？
2. FN 是否减少？
3. Recall 是否提高？
4. FP 是否也随之增加？
5. Precision 是否下降？
6. Avg IoU 是否变化？
```

## 2. 实验设置

固定条件：

```text
model = yolo11s.pt
conf = 0.25
nms_iou = 0.70
eval_iou = 0.50
dataset = MOT17-02-SDP
```

只改变：

```text
imgsz
```

实验组：

```text
imgsz = 640
imgsz = 960
imgsz = 1280
```

## 3. 脚本修改

为了支持输入尺寸实验，对 `src/detect.py` 增加了：

```text
--imgsz
```

核心参数：

```python
parser.add_argument(
    "--imgsz",
    type=int,
    default=640,
    help="Inference image size.",
)
```

并传入 YOLO：

```python
results = model.predict(
    source=source,
    conf=conf,
    iou=iou,
    imgsz=imgsz,
    ...
)
```

## 4. 实验结果

| imgsz | all detections | person predictions | GT | TP | FP | FN | Precision | Recall | Avg IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640 | 9618 | 6530 | 18581 | 5176 | 1354 | 13405 | 0.7926 | 0.2786 | 0.8213 |
| 960 | 13151 | 8815 | 18581 | 6771 | 2044 | 11810 | 0.7681 | 0.3644 | 0.8067 |
| 1280 | 14833 | 10388 | 18581 | 7599 | 2789 | 10982 | 0.7315 | 0.4090 | 0.8020 |

说明：

```text
all detections 是全部类别预测框数量。
person predictions 是参与 MOT17 person GT 评估的预测框数量。
```

## 5. 结果分析

### 5.1 Recall 明显提高

随着 imgsz 增大：

```text
imgsz 640  -> Recall = 0.2786
imgsz 960  -> Recall = 0.3644
imgsz 1280 -> Recall = 0.4090
```

TP 也持续增加：

```text
5176 -> 6771 -> 7599
```

FN 持续减少：

```text
13405 -> 11810 -> 10982
```

结论：

```text
更大的输入尺寸确实帮助模型找回了更多 MOT17 行人目标。
这说明 MOT17 中存在大量小目标、远距离目标或密集遮挡目标，较高分辨率有助于检测。
```

### 5.2 Precision 逐步下降

随着 imgsz 增大：

```text
imgsz 640  -> Precision = 0.7926
imgsz 960  -> Precision = 0.7681
imgsz 1280 -> Precision = 0.7315
```

FP 也持续增加：

```text
1354 -> 2044 -> 2789
```

结论：

```text
更大的 imgsz 会让模型看到更多细节，也会产生更多候选框。
其中一部分变成 TP，但也有一部分变成 FP。
```

因此，增大 `imgsz` 的效果不是单纯变好，而是：

```text
Recall 提高，Precision 下降。
```

### 5.3 Avg IoU 略微下降

Avg IoU：

```text
imgsz 640  -> 0.8213
imgsz 960  -> 0.8067
imgsz 1280 -> 0.8020
```

这说明：

```text
高 imgsz 找回了更多目标，但新增加的 TP 可能包含更难检测的小目标或遮挡目标，
这些框的定位质量不一定比原来更高。
```

因此 Avg IoU 有轻微下降是可以理解的。

## 6. 参数取舍

如果目标是减少误检、保持较高 Precision：

```text
imgsz = 640
```

更合适。

如果目标是提高 Recall、减少漏检：

```text
imgsz = 960 或 1280
```

更合适。

当前结果中：

```text
imgsz = 1280
```

Recall 最高，但 FP 也最多。

```text
imgsz = 960
```

在 Recall 提升和 Precision 保持之间更折中：

```text
Recall:    0.2786 -> 0.3644
Precision: 0.7926 -> 0.7681
```

## 7. Day12 结论

本次实验说明：

> 在 MOT17 密集行人场景中，增大输入尺寸可以显著提升 YOLO11s 的 Recall，说明更高分辨率有助于检测小目标、远距离目标和密集行人。但同时 FP 也会增加，导致 Precision 下降。

综合来看：

```text
imgsz=960 是当前比较合理的折中点；
imgsz=1280 适合更重视 Recall 的场景；
imgsz=640 适合更重视 Precision 和推理成本的场景。
```

## 8. 下一步

Day13 建议进入：

```text
模型规模与输入尺寸的联合对比
```

可以比较：

```text
yolo11n + imgsz640
yolo11s + imgsz640
yolo11s + imgsz960
yolo11s + imgsz1280
```

并加入推理速度或总运行时间，形成更完整的工程取舍表：

```text
速度 / Precision / Recall / Avg IoU / 模型大小
```
