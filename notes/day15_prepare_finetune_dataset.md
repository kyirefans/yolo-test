# Day15：准备 MOT17 YOLO 微调数据集

## 1. 学习目标

Day15 正式进入微调前的数据准备阶段。

目标是把 MOT17 的原始标注：

```text
gt/gt.txt
```

转换成 YOLO 训练需要的格式：

```text
class_id x_center y_center width height
```

并生成标准 YOLO 数据集目录。

## 2. 为什么要准备微调数据集

当前使用的模型：

```text
yolo11n.pt
yolo11s.pt
```

是 COCO 预训练模型。它们具备通用 person 检测能力，但 MOT17 的场景具有明显 domain gap：

```text
1. 监控视角；
2. 密集行人；
3. 远距离小目标；
4. 遮挡严重；
5. 背景相对固定；
6. 行人尺度分布和 COCO 不同。
```

因此模型在 MOT17 上 Recall 偏低。微调的目标是让模型适应 MOT17 行人场景，让更多 FN 变成 TP。

## 3. 新增脚本

新增脚本：

```text
scripts/prepare_mot17_yolo_dataset.py
```

功能：

```text
1. 读取 MOT17-02-SDP/seqinfo.ini；
2. 读取 MOT17-02-SDP/gt/gt.txt；
3. 只保留正式 pedestrian 标注；
4. 转换为 YOLO label 格式；
5. 复制图像到 train / val；
6. 生成 data.yaml；
7. 输出 prepare_summary.json。
```

## 4. MOT17 GT 到 YOLO 格式的转换

MOT17 GT 格式：

```text
frame_id, track_id, x, y, w, h, mark, class_id, visibility
```

本次只保留：

```text
mark = 1
class_id = 1
```

也就是正式标注的 pedestrian。

YOLO 单类数据集只训练一个类别：

```text
0: person
```

所以所有 label 的类别编号都写成：

```text
0
```

坐标转换公式：

```text
x_center = (x + w / 2) / image_width
y_center = (y + h / 2) / image_height
width    = w / image_width
height   = h / image_height
```

MOT17-02-SDP 图像尺寸：

```text
width = 1920
height = 1080
```

## 5. 生成的数据集

运行命令：

```bash
python scripts/prepare_mot17_yolo_dataset.py \
  --mot-dir datasets/MOT17-02-SDP \
  --out datasets/mot17_yolo_small \
  --train-count 300 \
  --val-count 100 \
  --start-frame 1 \
  --stride 1 \
  --gt-class-id 1 \
  --min-visibility 0.0
```

输出目录：

```text
datasets/mot17_yolo_small/
```

目录结构：

```text
datasets/mot17_yolo_small/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
  prepare_summary.json
```

## 6. 数据集统计

本次生成结果：

| split | images | labels | boxes | empty labels |
| --- | ---: | ---: | ---: | ---: |
| train | 300 | 300 | 8668 | 0 |
| val | 100 | 100 | 3407 | 0 |

校验结果：

```text
train images 300 labels 300 boxes 8668 missing_labels 0 bad 0
val images 100 labels 100 boxes 3407 missing_labels 0 bad 0
```

说明：

```text
1. 每张图片都有对应 label 文件；
2. label 坐标均在 0~1 范围内；
3. 类别编号均为 0；
4. 没有发现格式错误。
```

## 7. data.yaml

生成的配置文件：

```text
datasets/mot17_yolo_small/data.yaml
```

内容：

```yaml
path: datasets/mot17_yolo_small
train: images/train
val: images/val

names:
  0: person
```

这个文件会在 YOLO 训练时使用。

## 8. 当前数据集的限制

本次是小规模微调数据集，主要用于跑通流程。

限制包括：

```text
1. 只使用 MOT17-02-SDP 一个序列；
2. train 和 val 来自同一个视频序列；
3. 帧之间时间相关性很强；
4. 数据量较小；
5. 当前没有做可见度过滤；
6. 当前只训练 person 一个类别。
```

因此它不适合直接作为最终高质量训练集，但非常适合：

```text
1. 验证 YOLO train 流程；
2. 检查数据格式；
3. 观察微调是否能提高 MOT17 场景 Recall；
4. 学习微调前后对比方法。
```

## 9. 下一步

Day16 建议进入：

```text
第一次 YOLO 微调实验
```

建议从小训练开始：

```bash
yolo detect train \
  model=yolo11s.pt \
  data=datasets/mot17_yolo_small/data.yaml \
  epochs=5 \
  imgsz=960 \
  batch=4 \
  project=experiments/day16_train_mot17_yolo_small \
  name=yolo11s_imgsz960_epochs5
```

训练完成后，需要比较：

```text
1. 预训练 yolo11s.pt；
2. 微调后的 best.pt；
3. Precision / Recall / TP / FP / FN；
4. 典型帧可视化效果。
```
