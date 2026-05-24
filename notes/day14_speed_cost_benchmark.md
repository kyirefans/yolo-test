# Day14：推理速度与工程成本评估

## 1. 实验目的

前面的实验主要关注检测效果：

```text
Precision
Recall
TP / FP / FN
Avg IoU
```

Day14 开始补充工程维度：

```text
模型大小
推理总耗时
FPS
效果与速度的取舍
```

核心问题：

> Day13 推荐的 `yolo11s + imgsz960` 是否仍然是效果和速度之间比较平衡的配置？

## 2. 实验设置

数据：

```text
MOT17-02-SDP
600 frames
```

测速脚本：

```text
scripts/benchmark_inference.py
```

固定参数：

```text
conf = 0.25
nms_iou = 0.70
```

对比配置：

```text
yolo11n + imgsz640
yolo11s + imgsz640
yolo11s + imgsz960
yolo11s + imgsz1280
```

测速输出：

```text
experiments/day14_benchmark/
```

## 3. 综合结果

| config | model size MB | imgsz | Precision | Recall | Avg IoU | FPS | time sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n_imgsz640 | 5.35 | 640 | 0.8009 | 0.2471 | 0.8098 | 13.98 | 42.913 |
| yolo11s_imgsz640 | 18.42 | 640 | 0.7926 | 0.2786 | 0.8213 | 19.40 | 30.925 |
| yolo11s_imgsz960 | 18.42 | 960 | 0.7681 | 0.3644 | 0.8067 | 21.25 | 28.232 |
| yolo11s_imgsz1280 | 18.42 | 1280 | 0.7315 | 0.4090 | 0.8020 | 12.44 | 48.225 |

## 4. 模型大小分析

模型文件大小：

```text
yolo11n.pt = 5.35 MB
yolo11s.pt = 18.42 MB
```

说明：

```text
yolo11s 约为 yolo11n 的 3.4 倍大小。
```

但从 Day11 结果看，yolo11s 带来了更高 Recall：

```text
yolo11n + imgsz640 Recall = 0.2471
yolo11s + imgsz640 Recall = 0.2786
```

因此，模型变大确实带来了一定检测收益。

## 5. FPS 结果分析

当前单次测速结果：

```text
yolo11n + imgsz640  -> 13.98 FPS
yolo11s + imgsz640  -> 19.40 FPS
yolo11s + imgsz960  -> 21.25 FPS
yolo11s + imgsz1280 -> 12.44 FPS
```

这里有一个需要注意的现象：

```text
yolo11s + imgsz960 的 FPS 高于 yolo11s + imgsz640。
```

这不符合通常直觉，因为更大的输入尺寸一般会带来更高计算成本。

可能原因：

```text
1. 单次测速受到模型加载、缓存、预热影响；
2. 不同实验运行时系统负载不同；
3. 当前脚本记录的是端到端 predict 时间，不是严格模型前向时间；
4. Ultralytics 内部预处理、batch、缓存等机制可能影响单次结果。
```

因此，Day14 的 FPS 结果可以作为初步参考，但不应作为最终性能结论。后续如果要严谨比较速度，应加入：

```text
1. warmup；
2. 多次 repeat；
3. 固定运行顺序；
4. 记录平均值和标准差；
5. 区分预处理、推理、后处理耗时。
```

## 6. 效果与速度的取舍

### 6.1 yolo11n + imgsz640

优点：

```text
模型最小，文件只有 5.35 MB。
Precision 最高，为 0.8009。
```

缺点：

```text
Recall 只有 0.2471，漏检较多。
```

适合：

```text
模型体积优先、误检控制优先的场景。
```

### 6.2 yolo11s + imgsz640

优点：

```text
Recall 比 yolo11n 提高到 0.2786。
Avg IoU 最高，为 0.8213。
Precision 仍接近 0.79。
```

缺点：

```text
模型大小增加到 18.42 MB。
Recall 仍然偏低。
```

适合：

```text
希望在不提高 imgsz 的情况下获得稳定提升的场景。
```

### 6.3 yolo11s + imgsz960

优点：

```text
Recall 提高到 0.3644。
Precision 仍保持 0.7681。
当前测速 FPS 为 21.25。
```

缺点：

```text
FP 比 640 更多。
速度结果需要复测确认。
```

适合：

```text
当前阶段最平衡的实验 baseline。
```

### 6.4 yolo11s + imgsz1280

优点：

```text
Recall 最高，为 0.4090。
TP 最多，为 7599。
```

缺点：

```text
FP 最多，为 2789。
Precision 下降到 0.7315。
FPS 最低，为 12.44。
```

适合：

```text
更重视召回、可以接受更多误检和更高计算成本的场景。
```

## 7. 当前推荐配置

综合 Precision、Recall 和当前速度结果，仍然推荐：

```text
model = yolo11s.pt
imgsz = 960
conf = 0.25
nms_iou = 0.70
```

理由：

```text
1. Recall 明显高于 640；
2. Precision 仍在可接受范围；
3. FP 少于 imgsz1280；
4. 工程成本比 1280 更低；
5. 当前测速结果没有显示明显速度劣势。
```

但需要补充说明：

```text
速度结论需要通过多次重复 benchmark 进一步确认。
```

## 8. Day14 结论

本次实验把效果指标和工程指标放在了一起。

当前结论：

> `yolo11s + imgsz960` 仍然是当前最推荐的 MOT17 baseline。它在 Recall 上明显优于 640，在 Precision 上明显优于 1280，同时工程成本看起来可控。

不过，当前测速是初步结果。后续如果要严谨做工程部署判断，需要改进 benchmark 方法，加入 warmup 和 repeat。

## 9. 下一步

Day15 建议进入：

```text
改进 benchmark 脚本：加入 warmup / repeat / 平均 FPS
```

或者进入：

```text
准备小规模微调数据集
```

如果继续工程评估路线，建议先做 Day15 benchmark 改进，这样速度结论更可靠。
