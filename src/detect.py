from pathlib import Path
import argparse
import json

from ultralytics import YOLO


def run_detection(
    source: str,
    model_path: str = "yolo11n.pt",
    output_dir: str = "outputs",
    conf: float = 0.25,
    name: str = "detect_result",
    save_json: bool = True,
    iou: float = 0.7,
):
    """
    Run object detection on an image or video.

    Args:
        source: Path to image/video or URL.
        model_path: YOLO model weight path.
        output_dir: Directory to save results.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        name: Experiment/output name.
        save_json: Whether to save detection results as JSON.
    """
    if not source.startswith("http"):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project=output_dir,
        name=name,
        exist_ok=True,
        iou=iou,
    )

    all_results = []

    for result in results:
        image_result = {
            "source": str(result.path),
            "detections": [],
        }

        print("\nDetected objects:")

        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            all_results.append(image_result)
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detection = {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box_xyxy": [round(x, 2) for x in xyxy],
            }

            image_result["detections"].append(detection)
            print(detection)

        all_results.append(image_result)

    if save_json:
        json_dir = Path(output_dir) / name
        json_dir.mkdir(parents=True, exist_ok=True)

        json_path = json_dir / "detections.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\nDetection results saved to: {json_path}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on an image or video."
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path or URL of image/video source.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model weight path.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="detect_result",
        help="Output experiment name.",
    )

    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON saving.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_detection(
        source=args.source,
        model_path=args.model,
        output_dir=args.output,
        conf=args.conf,
        name=args.name,
        save_json=not args.no_json,
        iou=args.iou,
    )
