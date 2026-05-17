# scripts/validate_coco128.py

from ultralytics import YOLO
from pathlib import Path
import json


def main():
    output_dir = Path("experiments/day5_coco128_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")

    results = model.val(
        data="coco128.yaml",
        imgsz=640,
        plots=True,
        verbose=True,
    )

    print("\n=== Dataset-level metrics ===")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP75:    {results.box.map75:.4f}")

    image_metrics = results.box.image_metrics

    print("\n=== Per-image metrics: first 5 images ===")
    for idx, (image_name, metrics) in enumerate(image_metrics.items()):
        if idx >= 5:
            break

        print(f"\nImage: {image_name}")
        print(f"  TP: {metrics['tp']}")
        print(f"  FP: {metrics['fp']}")
        print(f"  FN: {metrics['fn']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

    summary = {
        "map50_95": float(results.box.map),
        "map50": float(results.box.map50),
        "map75": float(results.box.map75),
        "image_metrics": image_metrics,
    }

    summary_path = output_dir / "coco128_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nValidation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()