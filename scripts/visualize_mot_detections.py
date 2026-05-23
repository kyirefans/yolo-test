from pathlib import Path
import argparse

import cv2

from evaluate_mot_detections import (
    box_iou,
    load_mot_ground_truth,
    load_predictions,
    parse_gt_class_ids,
)


COLORS = {
    "gt": (0, 180, 0),
    "tp": (255, 120, 0),
    "fp": (0, 0, 255),
    "fn": (0, 220, 255),
    "text_bg": (20, 20, 20),
    "text": (255, 255, 255),
}


def parse_frames(value):
    frames = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        frames.append(int(item))
    return frames


def image_path_for_frame(images_dir, frame_id, image_ext):
    return Path(images_dir) / f"{frame_id:06d}{image_ext}"


def match_predictions(predictions, ground_truths, iou_threshold):
    predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    matched_gt = set()
    matched_predictions = []
    fp_predictions = []

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
            matched_predictions.append(
                {
                    "prediction": pred,
                    "gt_index": best_gt_index,
                    "iou": best_iou,
                }
            )
        else:
            fp_predictions.append(pred)

    fn_gt = [
        {"gt_index": gt_index, "gt": gt}
        for gt_index, gt in enumerate(ground_truths)
        if gt_index not in matched_gt
    ]

    return matched_predictions, fp_predictions, fn_gt


def draw_box(image, box, color, label, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        text_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness,
        )
        y_text = max(0, y1 - text_h - baseline - 3)
        cv2.rectangle(
            image,
            (x1, y_text),
            (x1 + text_w + 4, y_text + text_h + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 2, y_text + text_h + 1),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def draw_panel(image, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    line_height = 26
    width = 640
    height = 16 + line_height * len(lines)

    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), COLORS["text_bg"], -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    y = 34
    for line in lines:
        cv2.putText(
            image,
            line,
            (18, y),
            font,
            font_scale,
            COLORS["text"],
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def visualize_frame(
    frame_id,
    image_path,
    predictions,
    ground_truths,
    output_path,
    iou_threshold,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    matched_predictions, fp_predictions, fn_gt = match_predictions(
        predictions,
        ground_truths,
        iou_threshold,
    )

    for match in matched_predictions:
        gt = ground_truths[match["gt_index"]]
        draw_box(image, gt["box_xyxy"], COLORS["gt"], "GT", thickness=1)
        pred = match["prediction"]
        draw_box(
            image,
            pred["box_xyxy"],
            COLORS["tp"],
            f"TP {pred['confidence']:.2f} IoU {match['iou']:.2f}",
            thickness=2,
        )

    for pred in fp_predictions:
        draw_box(
            image,
            pred["box_xyxy"],
            COLORS["fp"],
            f"FP {pred['confidence']:.2f}",
            thickness=2,
        )

    for item in fn_gt:
        draw_box(image, item["gt"]["box_xyxy"], COLORS["fn"], "FN", thickness=2)

    tp = len(matched_predictions)
    fp = len(fp_predictions)
    fn = len(fn_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    draw_panel(
        image,
        [
            f"Frame {frame_id:06d}",
            f"TP={tp} FP={fp} FN={fn} Precision={precision:.3f} Recall={recall:.3f}",
            "GT=green, TP=blue, FP=red, FN=yellow",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

    return {
        "frame_id": frame_id,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "output_path": str(output_path),
    }


def visualize_mot_detections(
    images_dir,
    detections_json,
    gt_path,
    output_dir,
    frame_ids,
    image_ext,
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

    results = []
    for frame_id in frame_ids:
        image_path = image_path_for_frame(images_dir, frame_id, image_ext)
        output_path = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
        result = visualize_frame(
            frame_id=frame_id,
            image_path=image_path,
            predictions=pred_by_frame.get(frame_id, []),
            ground_truths=gt_by_frame.get(frame_id, []),
            output_path=output_path,
            iou_threshold=iou_threshold,
        )
        results.append(result)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO detections against MOTChallenge-style GT."
    )
    parser.add_argument("--images-dir", required=True, help="Path to MOT img1 directory.")
    parser.add_argument("--json", required=True, help="Path to detections.json.")
    parser.add_argument("--gt", required=True, help="Path to MOT gt.txt.")
    parser.add_argument("--out", required=True, help="Output directory for visualization images.")
    parser.add_argument("--frames", required=True, help="Comma-separated frame ids, e.g. 16,345,371.")
    parser.add_argument("--image-ext", default=".jpg", help="Image extension used by the MOT sequence.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="Evaluation IoU threshold.")
    parser.add_argument("--class-name", default="person", help="YOLO class name to visualize.")
    parser.add_argument("--gt-class-ids", default="1", help="Comma-separated MOT class ids, or 'all'.")
    parser.add_argument("--min-visibility", type=float, default=0.0, help="Minimum GT visibility.")
    parser.add_argument("--include-unmarked", action="store_true", help="Include unmarked GT rows.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gt_class_ids = parse_gt_class_ids(args.gt_class_ids)
    class_name = args.class_name if args.class_name else None

    results = visualize_mot_detections(
        images_dir=args.images_dir,
        detections_json=args.json,
        gt_path=args.gt,
        output_dir=args.out,
        frame_ids=parse_frames(args.frames),
        image_ext=args.image_ext,
        iou_threshold=args.iou_thres,
        class_name=class_name,
        gt_class_ids=gt_class_ids,
        min_visibility=args.min_visibility,
        marked_only=not args.include_unmarked,
    )

    print("\nVisualization outputs")
    print("=====================")
    for result in results:
        print(
            f"frame={result['frame_id']} "
            f"TP={result['tp']} FP={result['fp']} FN={result['fn']} "
            f"P={result['precision']:.3f} R={result['recall']:.3f} "
            f"path={result['output_path']}"
        )
