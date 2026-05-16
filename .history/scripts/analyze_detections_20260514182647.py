from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json


def calculate_box_area(box_xyxy):
    """
    Calculate bounding box area from [x1, y1, x2, y2].
    """
    if len(box_xyxy) != 4:
        return 0.0

    x1, y1, x2, y2 = box_xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def load_detection_json(json_path: str):
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_detections(results):
    """
    Analyze detection results exported by src/detect.py.
    """
    class_counter = Counter()
    confidence_by_class = defaultdict(list)
    area_by_class = defaultdict(list)

    total_sources = len(results)
    total_detections = 0

    for item in results:
        detections = item.get("detections", [])
        total_detections += len(detections)

        for det in detections:
            class_name = det.get("class_name", "unknown")
            confidence = float(det.get("confidence", 0.0))
            box_xyxy = det.get("box_xyxy", [])

            class_counter[class_name] += 1
            confidence_by_class[class_name].append(confidence)
            area_by_class[class_name].append(calculate_box_area(box_xyxy))

    per_class_stats = []

    for class_name, count in class_counter.most_common():
        confidences = confidence_by_class[class_name]
        areas = area_by_class[class_name]

        avg_confidence = sum(confidences) / len(confidences)
        avg_area = sum(areas) / len(areas)

        per_class_stats.append(
            {
                "class_name": class_name,
                "count": count,
                "avg_confidence": round(avg_confidence, 4),
                "max_confidence": round(max(confidences), 4),
                "min_confidence": round(min(confidences), 4),
                "avg_box_area": round(avg_area, 2),
            }
        )

    summary = {
        "total_sources": total_sources,
        "total_detections": total_detections,
        "class_counts": dict(class_counter),
        "per_class_stats": per_class_stats,
    }

    return summary


def save_summary(summary, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_path / "summary.json"
    summary_csv_path = output_path / "summary.csv"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "class_name",
            "count",
            "avg_confidence",
            "max_confidence",
            "min_confidence",
            "avg_box_area",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["per_class_stats"])

    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Summary CSV saved to: {summary_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze YOLO detection JSON results."
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to detections.json.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="experiments/day4_detection_stats",
        help="Output directory for analysis summary.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = load_detection_json(args.json)
    summary = analyze_detections(results)

    print("\nDetection Summary")
    print("=================")
    print(f"Total sources: {summary['total_sources']}")
    print(f"Total detections: {summary['total_detections']}")

    print("\nPer-class statistics:")
    for item in summary["per_class_stats"]:
        print(item)

    save_summary(summary, args.out)