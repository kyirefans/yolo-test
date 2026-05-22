# Day7 学习总结：从 YOLO 标签理解验证结果

## 1. 今天的核心问题

Day6 学的是 PR Curve、AP、mAP 这些评价指标。  
Day7 开始把指标往下拆，理解它们最底层的数据来源：

```text
YOLO 标签文件
-> GT 真实框
-> 预测框与 GT 计算 IoU
-> 根据 evaluation IoU threshold 判断 TP / FP / FN
-> 计算 Precision / Recall
-> 汇总成 AP / mAP
```

今天重点是：看懂一个 YOLO 标注文件如何参与验证结果计算。

## 2. YOLO 标签文件格式

当前例子：

```text
datasets/coco128/labels/train2017/000000000357.txt
```

YOLO 标签文件中，每一行代表一个真实目标，也就是一个 GT。

格式为：

```text
class_id x_center y_center width height
```

其中：

- `class_id`：类别编号。
- `x_center`：目标框中心点 x 坐标，归一化到 `0~1`。
- `y_center`：目标框中心点 y 坐标，归一化到 `0~1`。
- `width`：目标框宽度，归一化到 `0~1`。
- `height`：目标框高度，归一化到 `0~1`。

注意：这些坐标不是像素坐标，而是归一化坐标。

## 3. label 文件行数与 GT 数量

`000000000357.txt` 一共有 17 行，所以这张图片有：

```text
GT 总数 = 17
```

这 17 个 GT 会成为 Recall 的分母。

如果验证结果中：

```text
TP = 11
FN = 6
```

那么：

```text
TP + FN = 11 + 6 = 17
```

刚好对应这个 label 文件中的 17 个真实目标。

## 4. 对应验证结果

当前验证结果中，图像 `000000000357.jpg` 的统计为：

```text
TP = 11
FP = 197
FN = 6
Precision = 0.0529
Recall = 0.6471
```

Precision：

```text
Precision = TP / (TP + FP)
          = 11 / (11 + 197)
          = 11 / 208
          = 0.0529
```

Recall：

```text
Recall = TP / (TP + FN)
       = 11 / (11 + 6)
       = 11 / 17
       = 0.6471
```

解释：

```text
模型找回了 17 个真实目标中的 11 个，所以 Recall 为 0.6471。
但模型同时产生了 197 个错误预测框，所以 Precision 很低。
```

实验语言可以写成：

> 对于图像 `000000000357.jpg`，标注文件中共有 17 个真实目标。验证结果显示模型匹配到了其中 11 个目标，漏检 6 个，因此 Recall 为 0.6471；但同时产生了 197 个 FP，导致 Precision 仅为 0.0529，说明该图像上误检较多。

## 5. YOLO 坐标转像素坐标

YOLO 标注格式是：

```text
class_id x_center y_center width height
```

其中坐标是归一化值。  
如果图片尺寸为：

```text
image_width = W
image_height = H
```

那么转换为像素中心点和宽高：

```text
x_center_px = x_center * W
y_center_px = y_center * H
w_px        = width * W
h_px        = height * H
```

例子：

```text
图像尺寸：640 x 480
YOLO 标注：0 0.5 0.5 0.25 0.20
```

转换后：

```text
x_center_px = 0.5  * 640 = 320
y_center_px = 0.5  * 480 = 240
w_px        = 0.25 * 640 = 160
h_px        = 0.20 * 480 = 96
```

所以中心点和尺寸为：

```text
center = (320, 240)
size   = (160, 96)
```

## 6. 中心点格式转角点格式

IoU 计算通常使用角点格式：

```text
[x1, y1, x2, y2]
```

其中：

- `x1, y1`：左上角坐标。
- `x2, y2`：右下角坐标。

从中心点格式转换：

```text
x1 = x_center - w / 2
y1 = y_center - h / 2
x2 = x_center + w / 2
y2 = y_center + h / 2
```

接上面的例子：

```text
x_center = 320
y_center = 240
w = 160
h = 96
```

得到：

```text
x1 = 320 - 160 / 2 = 240
y1 = 240 - 96 / 2 = 192
x2 = 320 + 160 / 2 = 400
y2 = 240 + 96 / 2 = 288
```

角点框为：

```text
[240, 192, 400, 288]
```

## 7. IoU 计算

IoU 的定义：

```text
IoU = 交集面积 / 并集面积
```

对于两个框：

```text
A = [0, 0, 100, 100]
B = [50, 50, 150, 150]
```

注意：角点格式中的 `x2` / `y2` 是坐标，不是宽高。

A 面积：

```text
width_A  = 100 - 0 = 100
height_A = 100 - 0 = 100
area_A   = 100 * 100 = 10000
```

B 面积：

```text
width_B  = 150 - 50 = 100
height_B = 150 - 50 = 100
area_B   = 100 * 100 = 10000
```

交集区域：

```text
inter_x1 = max(0, 50) = 50
inter_y1 = max(0, 50) = 50
inter_x2 = min(100, 150) = 100
inter_y2 = min(100, 150) = 100
```

交集面积：

```text
inter_width  = 100 - 50 = 50
inter_height = 100 - 50 = 50
inter_area   = 50 * 50 = 2500
```

并集面积：

```text
union = area_A + area_B - inter_area
      = 10000 + 10000 - 2500
      = 17500
```

IoU：

```text
IoU = 2500 / 17500
    = 0.1429
```

## 8. IoU 与 TP / FP / FN 的关系

验证阶段会把预测框和 GT 框进行匹配。

如果：

```text
类别正确
Pred-GT IoU >= evaluation IoU threshold
```

则预测框可以被判为 TP。

如果预测框没有成功匹配到 GT，则是 FP。

如果某个 GT 没有被任何预测框成功匹配，则是 FN。

因此：

```text
Evaluation IoU Threshold 越高，TP 判定越严格。
原本在 IoU=0.5 下算 TP 的预测框，可能在 IoU=0.75 下变成 FP。
```

所以：

```text
mAP50 高于 mAP75 是正常现象。
```

真正需要关注的是二者差距：

- 差距小：模型不仅能找到目标，框也比较准。
- 差距大：模型能大致找到目标，但定位精度不足。

## 9. 当前 COCO128 结果解释

当前验证结果：

```text
mAP50    = 0.6699
mAP75    = 0.5390
mAP50-95 = 0.5026
```

解释：

```text
mAP50 高于 mAP75，说明模型在宽松 IoU=0.5 条件下具备一定目标检测能力；
但当 IoU 阈值提高到 0.75 时，部分预测框由于定位不够精确而不再满足 TP 条件。

mAP50 高于 mAP50-95，说明随着 IoU 阈值从 0.50 提高到 0.95，
模型性能逐步下降。这是正常现象，也说明模型定位精度还有提升空间。
```

规范实验分析：

> 本次使用 YOLO11n 在 COCO128 数据集上进行验证。实验结果显示，模型的 mAP50 为 0.6699，mAP75 为 0.5390，mAP50-95 为 0.5026。mAP50 高于 mAP75 和 mAP50-95，说明模型在较宽松的 IoU=0.5 条件下具备一定目标检测能力；但当 IoU 阈值提高后，性能明显下降，表明模型的定位精度仍有提升空间。

## 10. 今天必须记住的话

1. YOLO 标签文件中每一行代表一个 GT。
2. YOLO 标签坐标是归一化的中心点格式。
3. IoU 通常使用 `[x1, y1, x2, y2]` 角点格式计算。
4. 角点格式中的 `x2` / `y2` 是坐标，不是宽高。
5. Recall 的分母来自 GT 数量，也就是 `TP + FN`。
6. Precision 的分母来自预测为目标的框数量，也就是 `TP + FP`。
7. Evaluation IoU Threshold 提高后，TP 判定会更严格。
8. `mAP50 > mAP75` 是正常现象，差距可以反映定位精度问题。

## 11. 下一步

Day8 可以继续做：

```text
把 COCO128 验证结果整理成正式实验报告
```

重点包括：

- 读取并解释整体指标：`mAP50`、`mAP75`、`mAP50-95`。
- 选取若干单图像案例分析 TP / FP / FN。
- 解释为什么某些图像 Precision 很低。
- 为后续模型对比实验做准备，例如 `yolo11n` vs `yolo11s`、不同 `imgsz`、不同 NMS IoU。
