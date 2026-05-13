from pathlib import Path
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


if __name__ == "__main__":
    run_detection(
        source="httbus.jpg",
        model_path="yolo11n.pt",
        conf=0.25,
    )