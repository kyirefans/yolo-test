from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json


def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def parse_frame_id(source, fallback):
    stem = Path(source).stem
    if stem.isdigit():
        return int(stem)
    return fallback


def parse_gt_class_ids(value):
    if value.lower() == "all":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def load_mot_ground_truth(
    gt_path,
    gt_class_ids,
    marked_only=True,
    min_visibility=0.0,
):
    gt_by_frame = defaultdict(list)

    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue

            frame_id = int(float(row[0]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])

            mark = int(float(row[6])) if len(row) > 6 else 1
            class_id = int(float(row[7])) if len(row) > 7 else 1
            visibility = float(row[8]) if len(row) > 8 else 1.0

            if marked_only and mark != 1:
                continue
            if gt_class_ids is not None and class_id not in gt_class_ids:
                continue
            if visibility < min_visibility:
                continue

            gt_by_frame[frame_id].append(
                {
                    "track_id": int(float(row[1])),
                    "box_xyxy": xywh_to_xyxy(x, y, w, h),
                    "class_id": class_id,
                    "visibility": visibility,
                }
            )

    return gt_by_frame


def load_predictions(detections_json, class_name):
    pred_by_frame = defaultdict(list)
    data = json.loads(Path(detections_json).read_text(encoding="utf-8"))

    for index, item in enumerate(data, start=1):
        frame_id = parse_frame_id(item.get("source", ""), index)
        for det in item.get("detections", []):
            if class_name and det.get("class_name") != class_name:
                continue

            pred_by_frame[frame_id].append(
                {
                    "confidence": float(det.get("confidence", 0.0)),
                    "box_xyxy": [float(v) for v in det.get("box_xyxy", [])],
                    "class_name": det.get("class_name"),
                }
            )

    return pred_by_frame


def evaluate_frame(predictions, ground_truths, iou_threshold):
    predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    matched_gt = set()
    matches = []
    fp = 0

    for pred in predictions:
        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(ground_truths):
            if gt_index in matched_gt:
                continue
            iou = box_iou(pred["box_xyxy"], gt["box_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_index)
            matches.append(best_iou)
        else:
            fp += 1

    tp = len(matches)
    fn = len(ground_truths) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_predictions": len(predictions),
        "num_gt": len(ground_truths),
        "matched_ious": matches,
    }


def evaluate_mot_detections(
    detections_json,
    gt_path,
    iou_threshold,
    class_name,
    gt_class_ids,
    min_visibility,
    marked_only,
):
    pred_by_frame = load_predictions(detections_json, class_name)
    gt_by_frame = load_mot_ground_truth(
        gt_path=gt_path,
        gt_class_ids=gt_class_ids,
        marked_only=marked_only,
        min_visibility=min_visibility,
    )

    frame_ids = sorted(set(pred_by_frame) | set(gt_by_frame))
    frame_metrics = []
    all_matched_ious = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predictions = 0
    total_gt = 0

    for frame_id in frame_ids:
        metrics = evaluate_frame(
            pred_by_frame.get(frame_id, []),
            gt_by_frame.get(frame_id, []),
            iou_threshold,
        )

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_predictions += metrics["num_predictions"]
        total_gt += metrics["num_gt"]
        all_matched_ious.extend(metrics["matched_ious"])

        precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if metrics["tp"] + metrics["fp"] else 0.0
        recall = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if metrics["tp"] + metrics["fn"] else 0.0

        frame_metrics.append(
            {
                "frame_id": frame_id,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "num_predictions": metrics["num_predictions"],
                "num_gt": metrics["num_gt"],
                "precision": precision,
                "recall": recall,
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    avg_matched_iou = sum(all_matched_ious) / len(all_matched_ious) if all_matched_ious else 0.0

    return {
        "settings": {
            "detections_json": str(detections_json),
            "gt_path": str(gt_path),
            "iou_threshold": iou_threshold,
            "class_name": class_name,
            "gt_class_ids": sorted(gt_class_ids) if gt_class_ids is not None else "all",
            "min_visibility": min_visibility,
            "marked_only": marked_only,
        },
        "summary": {
            "num_frames": len(frame_ids),
            "total_predictions": total_predictions,
            "total_gt": total_gt,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "avg_matched_iou": avg_matched_iou,
        },
        "frame_metrics": frame_metrics,
    }


def save_results(results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "mot_eval_summary.json"
    frame_csv_path = output_path / "mot_eval_frames.csv"

    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(frame_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "frame_id",
            "tp",
            "fp",
            "fn",
            "num_predictions",
            "num_gt",
            "precision",
            "recall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results["frame_metrics"])

    print(f"Summary saved to: {summary_path}")
    print(f"Frame metrics saved to: {frame_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detections against MOTChallenge-style ground truth."
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to detections.json exported by src/detect.py.",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to MOT gt.txt.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for MOT evaluation results.",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="Evaluation IoU threshold for matching predictions to GT.",
    )
    parser.add_argument(
        "--class-name",
        default="person",
        help="YOLO class name to evaluate. Use an empty string to keep all classes.",
    )
    parser.add_argument(
        "--gt-class-ids",
        default="1",
        help="Comma-separated MOT GT class ids to evaluate, or 'all'. MOT pedestrian is usually 1.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.0,
        help="Minimum MOT visibility value to keep a GT box.",
    )
    parser.add_argument(
        "--include-unmarked",
        action="store_true",
        help="Include MOT GT rows whose mark flag is not 1.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gt_class_ids = parse_gt_class_ids(args.gt_class_ids)
    class_name = args.class_name if args.class_name else None

    results = evaluate_mot_detections(
        detections_json=args.json,
        gt_path=args.gt,
        iou_threshold=args.iou_thres,
        class_name=class_name,
        gt_class_ids=gt_class_ids,
        min_visibility=args.min_visibility,
        marked_only=not args.include_unmarked,
    )

    summary = results["summary"]
    print("\nMOT Detection Evaluation")
    print("========================")
    print(f"Frames:           {summary['num_frames']}")
    print(f"Predictions:      {summary['total_predictions']}")
    print(f"GT boxes:         {summary['total_gt']}")
    print(f"TP:               {summary['tp']}")
    print(f"FP:               {summary['fp']}")
    print(f"FN:               {summary['fn']}")
    print(f"Precision:        {summary['precision']:.4f}")
    print(f"Recall:           {summary['recall']:.4f}")
    print(f"Avg matched IoU:  {summary['avg_matched_iou']:.4f}")

    save_results(results, args.out)
