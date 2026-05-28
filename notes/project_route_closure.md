# 当前项目路线如何闭环

## 1. 闭环的定义

这个项目不是单纯学 YOLO，也不是单纯堆实验脚本。真正的闭环是：

```text
场景问题
-> 数据与标注
-> 模型推理或训练
-> 指标评估
-> 可视化误差分析
-> 实验报告
-> 下一轮改进决策
```

也就是说，每一阶段都要回答三个问题：

```text
1. 我现在解决的具体感知问题是什么？
2. 我用什么实验和指标证明模型表现？
3. 根据结果，下一步应该调参数、换模型、微调，还是进入下一个任务？
```

如果只有代码，没有指标和解释，项目没有闭环。  
如果只有理论，没有实验结果，学习也没有闭环。

## 2. 当前项目主线

当前项目路线可以概括为：

```text
目标检测基础
-> 检测指标理解
-> 误检/漏检分析
-> 参数实验
-> 模型与输入尺寸对比
-> 推理速度评估
-> MOT17 行人场景微调
-> 微调前后对比
-> 置信度阈值选择
-> 多目标跟踪
-> 行为识别
-> 视觉语言理解
-> 具身感知任务闭环
```

其中目前最完整的一条闭环是：

```text
MOT17 行人检测场景
-> 使用 YOLO11s 预训练模型检测
-> 发现 Recall 偏低
-> 将 MOT17 GT 转换为 YOLO 格式
-> 微调 YOLO11s
-> 对比微调前后指标
-> 调整 confidence threshold
-> 根据 Precision / Recall / F1 选择合适阈值
```

这说明项目已经从“跑模型”推进到了“针对场景改进模型”。

## 3. 已完成的检测阶段闭环

### 3.1 基础推理闭环

对应内容：

```text
Day1: 跑通 YOLO 推理
Day4: 分析 confidence threshold 对检测数量的影响
```

闭环逻辑：

```text
输入图片/视频
-> YOLO 输出检测框
-> 保存 detections.json
-> 统计类别、置信度、框面积
-> 理解 confidence threshold 如何影响输出数量
```

学到的核心点：

```text
confidence 越低，模型输出越多，Recall 可能上升，但 FP 也可能增加。
confidence 越高，模型输出越少，Precision 可能上升，但 FN 也可能增加。
```

### 3.2 指标理解闭环

对应内容：

```text
Day5: COCO128 validation
Day6: PR Curve / AP / mAP
Day7: YOLO label / IoU / TP / FP / FN
Day8: COCO128 validation report
```

闭环逻辑：

```text
预测框 + GT 框
-> IoU 匹配
-> 得到 TP / FP / FN
-> 计算 Precision / Recall
-> 形成 PR Curve
-> 计算 AP / mAP
-> 写实验报告解释模型问题
```

学到的核心点：

```text
mAP50 高于 mAP75 是正常现象。
Evaluation IoU threshold 越严格，原本算 TP 的框可能变成 FP 和 FN。
mAP50-95 比单个 mAP50 更能反映整体定位质量。
```

### 3.3 参数实验闭环

对应内容：

```text
Day9: confidence threshold 与 NMS IoU 实验
Day10: 检测结果可视化与错误分析
Day11: yolo11n vs yolo11s
Day12: imgsz 对检测效果的影响
Day13: 模型大小与输入尺寸的取舍
Day14: 推理速度和成本评估
```

闭环逻辑：

```text
固定数据集
-> 改一个实验变量
-> 重新推理和评估
-> 对比 Precision / Recall / Avg IoU / 推理速度
-> 找到效果、速度、成本之间的取舍
```

学到的核心点：

```text
模型越大，不一定在所有场景都绝对更好，但通常表达能力更强。
imgsz 越大，小目标更容易被保留，但速度和显存成本更高。
NMS IoU 影响重复框抑制强度。
confidence threshold 影响 Precision 和 Recall 的平衡。
```

## 4. 当前微调阶段闭环

### 4.1 为什么进入微调

预训练 YOLO 在 COCO 上学过 `person`，但没有专门针对 MOT17 这种场景训练：

```text
密集行人
远距离小目标
遮挡行人
监控视角
大量相似目标
```

所以模型能检测人，但对这个场景不够敏感，表现为：

```text
Recall 偏低
漏检较多
小目标和遮挡目标容易丢
```

微调的作用是：

```text
保留预训练模型已有的通用视觉能力
-> 用 MOT17 行人数据继续训练
-> 调整模型参数
-> 让模型更适应当前场景的数据分布
```

### 4.2 微调数据闭环

对应内容：

```text
Day15: 准备 MOT17 YOLO 格式数据集
Day16: 训练 YOLO11s 小规模微调模型
```

闭环逻辑：

```text
MOT17 原始 GT
-> 转换为 YOLO label 格式
-> 划分 train / val
-> 使用 YOLO train 进行微调
-> 生成 best.pt
-> 查看 results.csv 训练结果
```

这里必须转换为 YOLO 格式，是因为：

```text
评估脚本只需要读 MOT17 GT，所以原始格式够用。
训练 YOLO 时，Ultralytics dataloader 需要 YOLO label 格式，所以必须转换。
```

### 4.3 微调前后对比闭环

对应内容：

```text
Day17: 预训练模型 vs 微调模型
```

核心结果：

```text
预训练 YOLO11s:
Precision = 0.7681
Recall    = 0.3644
Avg IoU   = 0.8067

微调 YOLO11s:
Precision = 0.6646
Recall    = 0.6930
Avg IoU   = 0.7992
```

解释：

```text
微调后 Recall 大幅上升，说明模型找回了更多 MOT17 行人。
Precision 下降，说明模型也产生了更多 FP。
这不是微调失败，而是模型从保守检测变成更积极检测。
```

这一步的结论是：

```text
微调让模型适应了行人监控场景，但还需要通过阈值选择控制 FP。
```

### 4.4 阈值选择闭环

对应内容：

```text
Day18: 微调模型 confidence threshold 实验
```

实验结果：

| Confidence | Precision | Recall | F1 | Avg IoU |
|---:|---:|---:|---:|---:|
| 0.25 | 0.6646 | 0.6930 | 0.6785 | 0.7992 |
| 0.35 | 0.7929 | 0.6320 | 0.7044 | 0.8114 |
| 0.50 | 0.9269 | 0.5509 | 0.6912 | 0.8284 |
| 0.65 | 0.9852 | 0.4543 | 0.6220 | 0.8487 |

结论：

```text
conf=0.25: 更重视找全目标，Recall 高，但 FP 多。
conf=0.35: Precision 和 Recall 最平衡，F1 最高。
conf=0.50: Precision 很高，Recall 明显下降。
conf=0.65: 几乎只保留高置信度框，误检很少，但漏检明显增加。
```

当前检测阶段建议默认选择：

```text
conf=0.35
```

理由：

```text
它在当前 MOT17 检测实验中取得最高 F1，更适合作为后续 tracking 的输入。
```

## 5. 下一阶段如何闭环：从检测到跟踪

检测模型只回答：

```text
这一帧里有哪些人？
```

多目标跟踪要回答：

```text
同一个人如何在多帧之间保持同一个 ID？
```

下一阶段闭环应当是：

```text
YOLO 检测结果
-> 输入 SORT / ByteTrack
-> 生成 track id
-> 可视化轨迹
-> 评估 ID switch、轨迹连续性、漏跟踪
-> 分析检测错误如何影响跟踪
```

推荐 Day19-Day24：

```text
Day19: 写微调 confidence threshold 实验报告
Day20: 读 SORT，理解 Kalman Filter + Hungarian Matching
Day21: 跑通 YOLO + SORT/ByteTrack 跟踪
Day22: 读 ByteTrack，理解低分框为什么还能用于关联
Day23: 可视化 MOT17 跟踪结果
Day24: 写 tracking 阶段报告
```

跟踪阶段的核心问题会从：

```text
TP / FP / FN
```

扩展到：

```text
ID Switch
轨迹断裂
重复轨迹
短轨迹噪声
MOTA / IDF1 / HOTA
```

## 6. 再下一阶段如何闭环：从跟踪到行为识别

跟踪完成后，项目可以自然进入行为识别。

原因是行为识别通常不只看单帧，而是看时间片段：

```text
一个人的连续轨迹
-> 裁剪出视频片段
-> 输入行为识别模型
-> 输出动作类别
```

行为识别阶段闭环：

```text
跟踪得到 person track
-> 按 track 裁剪 clip
-> 跑 SlowFast / VideoMAE
-> 输出行为类别
-> 分析混淆动作
-> 写行为识别实验报告
```

这一阶段要重点理解：

```text
单帧检测解决“在哪里”
多目标跟踪解决“是谁”
行为识别解决“在做什么”
```

## 7. 视觉语言与开放词汇检测如何接入

当普通 YOLO 检测和跟踪闭环完成后，再进入视觉语言模型会更顺。

普通 YOLO 的限制是：

```text
只能检测训练类别中的目标。
```

Grounding DINO / YOLO-World 这类模型扩展为：

```text
输入文本提示
-> 检测开放词汇目标
```

例如：

```text
"person with backpack"
"helmet"
"red bag"
"person sitting"
```

开放词汇检测阶段闭环：

```text
文本 prompt
-> 开放词汇检测模型
-> 输出与文本相关的目标框
-> 和普通 YOLO 结果对比
-> 分析哪些目标必须依赖语言提示
```

这一步是从传统视觉任务走向具身感知的重要桥梁。

## 8. 最终具身感知闭环

具身感知不是只做检测，而是让视觉结果服务于任务决策。

最终可以形成：

```text
摄像头输入
-> 目标检测
-> 多目标跟踪
-> 行为识别
-> 视觉语言理解
-> 场景状态表示
-> 机器人/智能体决策
```

例如一个简单任务：

```text
判断前方是否有人
-> 判断这个人是否正在移动
-> 判断是否挡住路径
-> 决定等待、绕行或提醒
```

这时项目不再只是回答：

```text
模型准不准？
```

而是回答：

```text
感知结果能不能支持一个具体行动？
```

这就是 embodied perception 的闭环。

## 9. 论文阅读如何嵌入闭环

论文不要单独读，要跟项目问题绑定。

建议阅读顺序：

```text
YOLO / YOLOv8 or YOLO11 docs: 理解实时检测系统
SORT: 理解检测框如何变成轨迹
ByteTrack: 理解低置信度框如何帮助跟踪
DETR: 理解 transformer detection
Grounding DINO: 理解文本条件检测
YOLO-World: 理解实时开放词汇检测
SlowFast / VideoMAE: 理解行为识别
CLIP / LLaVA: 理解视觉语言对齐
```

每篇论文都按同一个模板输出笔记：

```text
1. 它解决什么问题？
2. 输入和输出是什么？
3. 核心模块是什么？
4. 它比之前方法强在哪里？
5. 它和当前项目哪个脚本或实验有关？
6. 我能不能用一个小实验验证它的思想？
```

论文阅读的闭环不是“读完论文”，而是：

```text
读论文
-> 找到项目中对应问题
-> 跑一个实验
-> 写一页解释
```

## 10. 当前最推荐的下一步

当前项目最自然的下一步是：

```text
Day19: 写 Day18 微调模型 confidence threshold 实验报告
```

报告应回答：

```text
1. 为什么微调后还要调 confidence threshold？
2. conf=0.25 / 0.35 / 0.50 / 0.65 各自适合什么场景？
3. 为什么 conf=0.35 是当前最平衡选择？
4. 这个阈值作为 tracking 输入可能有什么影响？
```

完成 Day19 后，就可以进入：

```text
Day20: SORT 论文与 tracking 基础
```

这样项目就会从：

```text
检测一个人
```

推进到：

```text
持续跟踪同一个人
```

这是从 object detection 进入 embodied perception 的关键一步。
