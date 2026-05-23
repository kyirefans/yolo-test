# Day13：模型规模与输入尺寸的综合取舍

## 1. 实验目的

Day11 比较了：

```text
yolo11n vs yolo11s
```

Day12 比较了：

```text
imgsz = 640 / 960 / 1280
```

Day13 的目标是把这些结果整合起来，回答一个更工程化的问题：

```text
在 MOT17 密集行人场景中，后续应该选择哪个配置作为 baseline？
```

这里不再只看单个指标，而是综合考虑：

```text
1. Precision
2. Recall
3. FP / FN
4. Avg IoU
5. 模型大小
6. 后续实验成本
```

## 2. 模型大小

当前本地模型文件：

| model | file size |
| --- | ---: |
| yolo11n.pt | 5.4M |
| yolo11s.pt | 19M |

说明：

```text
yolo11s 比 yolo11n 更大，通常具备更强表达能力，但推理成本也更高。
```

## 3. 配置对比表

固定条件：

```text
dataset = MOT17-02-SDP
conf = 0.25
nms_iou = 0.70
eval_iou = 0.50
```

对比结果：

| 配置 | predictions | TP | FP | FN | Precision | Recall | Avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n + imgsz640 | 5732 | 4591 | 1141 | 13990 | 0.8009 | 0.2471 | 0.8098 |
| yolo11s + imgsz640 | 6530 | 5176 | 1354 | 13405 | 0.7926 | 0.2786 | 0.8213 |
| yolo11s + imgsz960 | 8815 | 6771 | 2044 | 11810 | 0.7681 | 0.3644 | 0.8067 |
| yolo11s + imgsz1280 | 10388 | 7599 | 2789 | 10982 | 0.7315 | 0.4090 | 0.8020 |

## 4. 从 YOLO11n 到 YOLO11s 的变化

对比：

```text
yolo11n + imgsz640
yolo11s + imgsz640
```

变化：

```text
TP:        4591 -> 5176，增加 585
FN:       13990 -> 13405，减少 585
Recall:  0.2471 -> 0.2786，提高 0.0315
FP:        1141 -> 1354，增加 213
Precision:0.8009 -> 0.7926，略微下降
Avg IoU:  0.8098 -> 0.8213，略有提升
```

结论：

```text
更大的 yolo11s 能找回更多行人目标，Recall 提高；
Precision 只是小幅下降，整体收益是正向的。
```

因此，在 MOT17 当前任务中：

```text
yolo11s 优于 yolo11n。
```

## 5. 输入尺寸增大的变化

对比：

```text
yolo11s + imgsz640
yolo11s + imgsz960
yolo11s + imgsz1280
```

Recall：

```text
0.2786 -> 0.3644 -> 0.4090
```

TP：

```text
5176 -> 6771 -> 7599
```

FN：

```text
13405 -> 11810 -> 10982
```

说明：

```text
增大 imgsz 明显提高 Recall，减少漏检。
```

但 Precision：

```text
0.7926 -> 0.7681 -> 0.7315
```

FP：

```text
1354 -> 2044 -> 2789
```

说明：

```text
更高分辨率带来更多检测框，一部分成为 TP，但也带来更多 FP。
```

## 6. 不同需求下的配置选择

### 6.1 如果最重视 Precision

可以选择：

```text
yolo11n + imgsz640
```

或：

```text
yolo11s + imgsz640
```

二者 Precision 都接近 0.8。

但问题是：

```text
Recall 较低，漏检严重。
```

如果任务要求不能乱报，但可以接受漏检，这类配置更合适。

### 6.2 如果最重视 Recall

可以选择：

```text
yolo11s + imgsz1280
```

它的 Recall 最高：

```text
Recall = 0.4090
```

但代价是：

```text
FP = 2789
Precision = 0.7315
```

如果任务更关注“尽量找全行人”，这个配置更合适。

### 6.3 如果希望综合平衡

当前更推荐：

```text
yolo11s + imgsz960
```

原因：

```text
Recall 从 0.2786 提高到 0.3644，提升明显；
Precision 仍保持 0.7681，没有下降得太严重；
相比 imgsz1280，FP 更少，推理成本也更低。
```

## 7. 推荐 baseline

后续实验建议采用：

```text
model = yolo11s.pt
imgsz = 960
conf = 0.25
nms_iou = 0.70
eval_iou = 0.50
```

作为 MOT17 密集行人检测的当前 baseline。

推荐理由：

```text
1. 比 yolo11n + 640 有明显更高 Recall；
2. 比 yolo11s + 1280 有更少 FP；
3. Precision 仍保持在可接受水平；
4. 工程成本比 1280 更可控。
```

## 8. 当前瓶颈

即使使用：

```text
yolo11s + imgsz1280
```

Recall 也只有：

```text
0.4090
```

这说明当前瓶颈不只是模型规模或输入尺寸，还包括：

```text
1. MOT17 中行人密集；
2. 小目标和远距离目标较多；
3. 遮挡严重；
4. COCO 预训练模型没有针对 MOT17 场景专门优化；
5. 单纯提高 imgsz 会带来更多 FP。
```

因此，后续如果要继续提高性能，需要考虑：

```text
1. 针对 MOT17 或类似行人场景微调；
2. 使用更大模型；
3. 加入速度指标，做工程取舍；
4. 做小目标和遮挡目标的专项分析。
```

## 9. Day13 结论

本阶段实验说明：

```text
模型变大可以提高 Recall；
输入尺寸变大也可以提高 Recall；
但二者都会增加预测框数量，并可能引入更多 FP。
```

当前最合理的平衡点是：

```text
yolo11s + imgsz960
```

最终结论：

> 在 MOT17-02-SDP 密集行人场景中，`yolo11s + imgsz960` 是当前更适合作为后续实验 baseline 的配置。它相比 `yolo11n + imgsz640` 明显提高 Recall，同时 Precision 仍保持在相对可接受的水平。

## 10. 下一步

Day14 建议进入：

```text
推理速度与模型成本评估
```

需要补充：

```text
1. 每组配置推理总耗时；
2. FPS；
3. 模型文件大小；
4. Precision / Recall / FPS 综合表。
```

这样可以从“效果指标”进入“工程部署取舍”。
