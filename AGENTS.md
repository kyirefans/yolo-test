# AGENTS.md

## Project Overview

This repository is a learning-oriented embodied perception project. The current focus is object detection with YOLO, using code experiments to build intuition before moving into heavier theory, fine-tuning, and paper comparison.

The guiding learning loop is:

```text
run code -> inspect outputs -> understand metrics -> write notes -> design next experiment
```

The project is intentionally structured as a study workspace, not a production package. Codex should act as both a pragmatic coding agent and a learning assistant.

## Current Technical Scope

Current topic:

```text
Object Detection with YOLO / COCO128
```

Current model:

```text
yolo11n.pt
```

Current dataset:

```text
datasets/coco128
```

Project virtual environment:

```bash
conda activate embodied
```

Use this environment before running YOLO, validation, or experiment scripts.

Current concepts already studied:

- YOLO inference on image/video.
- Confidence threshold and its effect on output count.
- Detection JSON export.
- Class/count/confidence analysis.
- Ground truth labels in YOLO format.
- TP / FP / FN.
- Precision / Recall.
- PR Curve.
- AP / mAP.
- mAP50, mAP75, mAP50-95.
- YOLO normalized label coordinates.
- Conversion from YOLO center format to `[x1, y1, x2, y2]`.
- IoU and evaluation IoU threshold.

## Repository Layout

Important files and directories:

```text
src/detect.py
```

Runs YOLO inference on images or videos. It supports configurable confidence threshold and saves detection outputs as JSON.

```text
scripts/analyze_detections.py
```

Analyzes `detections.json` files exported by `src/detect.py`. It summarizes class counts, confidence statistics, and box area statistics.

```text
scripts/validate_coco128.py
```

Runs YOLO validation on COCO128 and saves dataset-level and per-image metrics.

```text
outputs/
```

Stores raw detection outputs for different inference settings, such as different confidence thresholds.

```text
experiments/
```

Stores experiment summaries and validation outputs.

Current important experiment files:

```text
experiments/day4_detection_stats/summary.json
experiments/day4_detection_stats_conf050/summary.json
experiments/day4_detection_stats_conf075/summary.json
experiments/day5_coco128_validation/coco128_validation_summary.json
```

```text
notes/
```

Stores learning notes. Current notes include:

```text
notes/day1_object_detection.md
notes/day6_learning_summary.md
notes/day7_learning_summary.md
```

```text
Study_Plan.md
```

Contains the broader learning strategy: run first, understand next, then improve and compare.

## Current Validation Results

The latest COCO128 validation summary is:

```text
mAP50    = 0.6699
mAP75    = 0.5390
mAP50-95 = 0.5026
```

Interpretation:

```text
The model has reasonable detection ability under the loose IoU=0.5 setting.
The lower mAP75 and mAP50-95 indicate that stricter localization requirements reduce performance.
This suggests the model can often find objects, but bounding box localization still has room to improve.
```

One important image-level case:

```text
Image: 000000000357.jpg
GT count = 17
TP = 11
FP = 197
FN = 6
Precision = 0.0529
Recall = 0.6471
```

Interpretation:

```text
The model recovers 11 of 17 ground-truth objects, but produces many false positives.
This image is useful for learning how low Precision and moderate Recall can happen together.
```

## Learning Plan

The project follows this object detection path:

```text
1. Run YOLO inference.
2. Test on custom images/videos.
3. Observe false positives and false negatives.
4. Understand confidence threshold, NMS, IoU, Precision/Recall, PR Curve, AP, and mAP.
5. Analyze validation results on COCO128.
6. Study Anchor / Anchor-free at a conceptual level.
7. Try small-scale fine-tuning.
8. Compare YOLO, DETR, and Grounding DINO at the paper/system level.
```

Completed or mostly completed:

```text
Day1: Run YOLO object detection.
Day4: Analyze confidence threshold effects.
Day5: Validate YOLO on COCO128.
Day6: Learn PR Curve, AP, and mAP.
Day7: Understand YOLO labels, GT boxes, IoU, TP/FP/FN.
```

Current stage:

```text
Day8: Build a COCO128 validation report and analyze false positives / false negatives.
```

Recommended next days:

```text
Day9: Study confidence threshold and NMS IoU effects on FP/FN.
Day10: Visualize detection results and inspect errors manually.
Day11: Compare yolo11n and yolo11s.
Day12: Compare different imgsz settings.
Day13: Prepare a small fine-tuning dataset.
Day14: Run a small fine-tuning experiment.
```

## Day8 Plan

Day8 should turn metric knowledge into experiment analysis.

Primary goal:

```text
Read COCO128 validation results and write a clear report explaining model strengths and weaknesses.
```

Tasks:

1. Read dataset-level metrics:

```text
map50_95
map50
map75
```

2. Explain what the gap between `mAP50`, `mAP75`, and `mAP50-95` says about localization quality.

3. Analyze image-level metrics in:

```text
experiments/day5_coco128_validation/coco128_validation_summary.json
```

4. Select representative cases:

```text
highest Precision images
lowest Precision images
highest Recall images
lowest Recall images
```

5. Write a report file, preferably:

```text
notes/day8_coco128_validation_report.md
```

Suggested report structure:

```markdown
# Day8: COCO128 Validation Report

## 1. Experiment Goal
## 2. Experiment Setup
## 3. Dataset-Level Metrics
## 4. Metric Interpretation
## 5. Image-Level Case Analysis
## 6. Current Model Problems
## 7. Next Experiments
```

## How Codex Should Work In This Repository

When helping the user learn:

- Prefer teaching through the current repository files and experiment outputs.
- Connect every concept to a concrete file, script, metric, or example.
- Ask short check questions after explaining a concept.
- Keep explanations step-by-step and avoid jumping directly into papers before the user has code intuition.
- Use Chinese for learning explanations unless the user asks otherwise.

When editing files:

- Do not overwrite existing notes unless explicitly asked.
- Prefer adding new daily summary files under `notes/`.
- Use clear names such as `day8_coco128_validation_report.md`.
- Preserve the user's uncommitted files.
- Avoid unrelated refactors.

When running commands:

- Use `rg`, `find`, `sed`, and small Python snippets for inspection.
- Do not assume packages are installed; check before running heavier scripts if needed.
- Prefer reading existing JSON summaries before rerunning expensive validation.

When writing code:

- Follow the simple script style already used in `src/` and `scripts/`.
- Keep scripts readable for a learner.
- Include concise comments only where they clarify non-obvious logic.
- Save experiment outputs under `experiments/dayX_*`.

## Useful Commands

Activate the project environment:

```bash
conda activate embodied
```

Run YOLO inference on a video:

```bash
python src/detect.py --source data/videos/test.mp4 --conf 0.25 --name video_conf_025
```

Analyze detection JSON:

```bash
python scripts/analyze_detections.py \
  --json outputs/video_conf_025/detections.json \
  --out experiments/day4_detection_stats
```

Validate on COCO128:

```bash
python scripts/validate_coco128.py
```

Inspect Day5 validation summary:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("experiments/day5_coco128_validation/coco128_validation_summary.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(data["map50"])
print(data["map75"])
print(data["map50_95"])
PY
```

## Important Learning Principles

Do not start by memorizing all formulas. The intended order is:

```text
1. Run the model.
2. Observe predictions.
3. Compare predictions with labels.
4. Understand errors.
5. Learn the metric formulas.
6. Write an experiment report.
7. Design the next experiment.
```

For object detection specifically:

```text
prediction boxes + ground-truth boxes
-> IoU matching
-> TP / FP / FN
-> Precision / Recall
-> PR Curve
-> AP
-> mAP
-> experiment report
```
