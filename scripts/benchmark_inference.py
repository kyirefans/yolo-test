from pathlib import Path
import argparse
import json
import time

from ultralytics import YOLO


def count_image_files(source):
    path = Path(source)
    if path.is_file():
        return 1
    if not path.is_dir():
        return None

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for item in path.iterdir() if item.suffix.lower() in image_exts)


def file_size_mb(path):
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.stat().st_size / (1024 * 1024)


def benchmark_inference(
    source,
    model_path,
    output_dir,
    name,
    conf,
    iou,
    imgsz,
    save_predictions,
):
    source_path = Path(source)
    if not source.startswith("http") and not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    frame_count = count_image_files(source)
    model = YOLO(model_path)

    start_time = time.perf_counter()
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=save_predictions,
        project=output_dir,
        name=name,
        exist_ok=True,
        verbose=False,
    )
    total_time = time.perf_counter() - start_time

    result_count = len(results)
    effective_frames = frame_count or result_count
    fps = effective_frames / total_time if total_time > 0 else 0.0
    total_detections = 0

    for result in results:
        if result.boxes is not None:
            total_detections += len(result.boxes)

    summary = {
        "source": str(source),
        "model": str(model_path),
        "model_size_mb": file_size_mb(model_path),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "frame_count": effective_frames,
        "result_count": result_count,
        "total_detections": total_detections,
        "total_time_sec": total_time,
        "fps": fps,
        "save_predictions": save_predictions,
    }

    benchmark_dir = Path(output_dir) / name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary, summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO inference speed on an image directory or media file."
    )
    parser.add_argument("--source", required=True, help="Path or URL of image/video source.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weight path.")
    parser.add_argument("--output", default="experiments/day14_benchmark", help="Output directory.")
    parser.add_argument("--name", required=True, help="Benchmark run name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save rendered prediction images. Disabled by default for cleaner timing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    summary, summary_path = benchmark_inference(
        source=args.source,
        model_path=args.model,
        output_dir=args.output,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_predictions=args.save_predictions,
    )

    print("\nInference Benchmark")
    print("===================")
    print(f"Model:            {summary['model']}")
    if summary["model_size_mb"] is not None:
        print(f"Model size:       {summary['model_size_mb']:.2f} MB")
    print(f"Source:           {summary['source']}")
    print(f"Frames:           {summary['frame_count']}")
    print(f"Detections:       {summary['total_detections']}")
    print(f"imgsz:            {summary['imgsz']}")
    print(f"conf:             {summary['conf']}")
    print(f"iou:              {summary['iou']}")
    print(f"Total time:       {summary['total_time_sec']:.3f} sec")
    print(f"FPS:              {summary['fps']:.2f}")
    print(f"Summary saved to: {summary_path}")
