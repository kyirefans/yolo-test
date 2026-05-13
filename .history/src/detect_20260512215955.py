from pathlib import Path
import argparse

from ultralytics import YOLO


def run_detection(
    source: str,
    model_path: str = "yolo11n.pt",
    output_dir: str = "outputs",
    conf: float = 0.25,
):
    """
    Run object detection on an image or video.

    Args:
        source: Path to image/video or URL.
        model_path: YOLO model weight path.
        output_dir: Directory to save results.
        conf: Confidence threshold.
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
        name="detect_result",
        exist_ok=True,
    )

    for result in results:
        print("\nDetected objects:")

        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            print(
                {
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "box": [round(x, 2) for x in xyxy],
                }
            )


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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_detection(
        source=args.source,
        model_path=args.model,
        output_dir=args.output,
        conf=args.conf,
    )