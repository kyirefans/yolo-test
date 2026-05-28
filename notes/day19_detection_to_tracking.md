# Day19: 从检测结果进入多目标跟踪

## 1. Day19 的学习目标

Day18 解决的问题是：

```text
微调后的 YOLO 模型应该选择什么 confidence threshold？
```

Day19 开始进入一个新的问题：

```text
检测框如何变成连续轨迹？
```

目标检测只回答单帧问题：

```text
这一帧里有哪些人？
```

多目标跟踪要回答跨帧问题：

```text
这些帧里的同一个人是谁？
```

所以从 Day19 开始，项目主线从 object detection 进入 multi-object tracking。

## 2. 检测和跟踪的区别

### 2.1 目标检测

输入：

```text
单张图片或单帧视频
```

输出：

```text
bounding box
class
confidence
```

例如：

```text
frame 1:
person, bbox=[x1,y1,x2,y2], conf=0.81
person, bbox=[x1,y1,x2,y2], conf=0.76
```

检测模型不关心：

```text
第 1 帧的 person A 和第 2 帧的 person A 是不是同一个人。
```

### 2.2 多目标跟踪

输入：

```text
连续多帧的检测结果
```

输出：

```text
track_id
bounding box
frame_id
```

例如：

```text
frame 1: track_id=3, bbox=...
frame 2: track_id=3, bbox=...
frame 3: track_id=3, bbox=...
```

跟踪模型关心的是：

```text
同一个人在连续帧中是否保持同一个 ID。
```

一句话区分：

```text
检测解决“在哪里”；
跟踪解决“是谁一直在那里”。
```

## 3. 为什么 tracking 不能只看 Precision

Day18 中 `conf=0.65` 的结果是：

```text
Precision = 0.9852
Recall    = 0.4543
```

这说明：

```text
保留下来的检测框几乎都很准；
但真实行人有一大半没有被检测出来。
```

对单帧检测来说，高 Precision 很有吸引力。

但对跟踪来说，过低 Recall 会导致：

```text
轨迹断裂
目标消失
重新出现时被分配新 ID
ID switch 增加
```

例如：

```text
frame 1: 检测到 person A
frame 2: 漏检 person A
frame 3: 漏检 person A
frame 4: 又检测到 person A
```

跟踪器可能会认为：

```text
原来的轨迹已经结束；
frame 4 出现的是一个新的目标。
```

这会导致：

```text
同一个人被分配两个不同的 track_id。
```

所以进入 tracking 后，指标关注点会变化。

检测阶段主要看：

```text
Precision / Recall / F1 / mAP
```

跟踪阶段还要看：

```text
ID Switch
轨迹连续性
轨迹断裂
MOTA
IDF1
HOTA
```

## 4. 为什么后续不直接选择 conf=0.65

`conf=0.65` 的问题不是不准，而是太保守。

它会过滤掉很多：

```text
远距离行人
小目标行人
遮挡行人
模糊行人
短暂低置信度行人
```

这些目标在单帧里可能不够可靠，但在连续视频里可能仍然有用。

对 tracking 来说，一个低置信度框可能可以帮助：

```text
延续已有轨迹
减少轨迹中断
避免重新分配 ID
```

所以跟踪任务不能简单使用：

```text
Precision 最高的阈值
```

而要选择：

```text
既保留足够目标，又不过度引入假框的阈值。
```

这就是 Day18 推荐 `conf=0.35` 的原因。

## 5. ByteTrack 的核心直觉

ByteTrack 的一个重要思想是：

```text
低置信度检测框不一定都是垃圾。
```

在拥挤行人场景中，低分框可能来自：

```text
遮挡
远距离
运动模糊
身体只露出一部分
光照变化
```

这些框如果直接丢弃，可能会让轨迹断掉。

ByteTrack 的思路可以简化理解为：

```text
先用高置信度检测框匹配可靠轨迹；
再用低置信度检测框尝试补充匹配已有轨迹。
```

也就是说，低分框不是用来随便生成新轨迹，而是用来：

```text
帮助已有轨迹不断开。
```

所以如果在 YOLO 推理阶段就把阈值设得过高，例如 `conf=0.65`，很多低分框已经被删除，ByteTrack 后面就没有机会再利用它们。

这就是为什么：

```text
conf=0.65 对单帧高精度检测友好；
但不一定对连续多目标跟踪友好。
```

## 6. SORT 的核心直觉

SORT 是一个经典的 tracking-by-detection 方法。

它的基本流程是：

```text
检测器输出当前帧的 boxes
-> Kalman Filter 预测已有轨迹在当前帧的位置
-> 计算预测框和检测框的匹配关系
-> Hungarian Matching 分配检测框给轨迹
-> 更新轨迹状态
```

可以先这样理解：

```text
Kalman Filter: 预测目标下一帧大概在哪里。
Hungarian Matching: 决定哪个检测框属于哪个已有轨迹。
```

SORT 不需要重新训练一个大模型，它主要依赖检测结果。

所以检测质量直接影响 tracking：

```text
FP 多 -> 可能产生假轨迹。
FN 多 -> 可能造成轨迹断裂。
框位置不稳定 -> 匹配容易失败。
```

这也是为什么 Day18 的阈值实验很重要。

## 7. 从 Day18 到 Day20 的衔接

Day18 得到的结论是：

```text
conf=0.35 是当前微调模型较平衡的检测阈值。
```

Day19 的结论是：

```text
这个阈值不是为了让单帧 Precision 最高，而是为了给后续 tracking 提供更稳定的输入。
```

下一步 Day20 要学习：

```text
SORT: Simple Online and Realtime Tracking
```

重点不是一开始推公式，而是先理解：

```text
1. tracking-by-detection 是什么？
2. Kalman Filter 在跟踪里预测什么？
3. Hungarian Matching 在跟踪里匹配什么？
4. 检测阶段的 FP / FN 会如何影响轨迹？
```

## 8. Day19 总结

Day19 最重要的一句话：

```text
检测阈值不是只为单帧检测指标服务，也要为后续 tracking 的轨迹连续性服务。
```

当前项目的连接关系是：

```text
YOLO 微调
-> confidence threshold 选择
-> 输出检测框
-> SORT / ByteTrack 关联检测框
-> 生成连续 track_id
```

所以 Day19 是项目从目标检测进入多目标跟踪的过渡日。

