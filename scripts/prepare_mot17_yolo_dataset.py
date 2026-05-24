from pathlib import Path
import argparse
import csv
import json
import shutil


def parse_seqinfo(seqinfo_path):
    info = {}
    with open(seqinfo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()

    return {
        "name": info.get("name"),
        "im_dir": info.get("imDir", "img1"),
        "seq_length": int(info["seqLength"]),
        "width": int(info["imWidth"]),
        "height": int(info["imHeight"]),
        "ext": info.get("imExt", ".jpg"),
    }


def load_mot_gt(
    gt_path,
    image_width,
    image_height,
    gt_class_id,
    min_visibility,
):
    labels_by_frame = {}

    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9:
                continue

            frame_id = int(float(row[0]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            mark = int(float(row[6]))
            class_id = int(float(row[7]))
            visibility = float(row[8])

            if mark != 1:
                continue
            if class_id != gt_class_id:
                continue
            if visibility < min_visibility:
                continue
            if w <= 0 or h <= 0:
                continue

            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            norm_w = w / image_width
            norm_h = h / image_height

            values = [x_center, y_center, norm_w, norm_h]
            values = [min(1.0, max(0.0, value)) for value in values]

            labels_by_frame.setdefault(frame_id, []).append(
                f"0 {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"
            )

    return labels_by_frame


def sample_frames(seq_length, train_count, val_count, start_frame, stride):
    candidates = list(range(start_frame, seq_length + 1, stride))
    needed = train_count + val_count
    if needed > len(candidates):
        raise ValueError(
            f"Requested {needed} frames, but only {len(candidates)} are available "
            f"with start_frame={start_frame}, stride={stride}."
        )

    selected = candidates[:needed]
    return selected[:train_count], selected[train_count:]


def write_yaml(output_dir):
    data_yaml = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: person\n"
    )
    path = Path(output_dir) / "data.yaml"
    path.write_text(data_yaml, encoding="utf-8")
    return path


def prepare_split(
    split_name,
    frame_ids,
    image_dir,
    output_dir,
    image_ext,
    labels_by_frame,
    allow_empty_labels,
):
    image_out_dir = Path(output_dir) / "images" / split_name
    label_out_dir = Path(output_dir) / "labels" / split_name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    written_labels = 0
    empty_labels = 0
    total_boxes = 0

    for frame_id in frame_ids:
        image_name = f"{frame_id:06d}{image_ext}"
        src_image = Path(image_dir) / image_name
        dst_image = image_out_dir / image_name
        dst_label = label_out_dir / f"{frame_id:06d}.txt"

        if not src_image.exists():
            raise FileNotFoundError(f"Image not found: {src_image}")

        labels = labels_by_frame.get(frame_id, [])
        if not labels and not allow_empty_labels:
            continue

        shutil.copy2(src_image, dst_image)
        dst_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

        copied_images += 1
        written_labels += 1
        total_boxes += len(labels)
        if not labels:
            empty_labels += 1

    return {
        "split": split_name,
        "images": copied_images,
        "label_files": written_labels,
        "boxes": total_boxes,
        "empty_label_files": empty_labels,
    }


def prepare_dataset(
    mot_dir,
    output_dir,
    train_count,
    val_count,
    start_frame,
    stride,
    gt_class_id,
    min_visibility,
    allow_empty_labels,
):
    mot_path = Path(mot_dir)
    output_path = Path(output_dir)
    seqinfo = parse_seqinfo(mot_path / "seqinfo.ini")
    image_dir = mot_path / seqinfo["im_dir"]
    gt_path = mot_path / "gt" / "gt.txt"

    labels_by_frame = load_mot_gt(
        gt_path=gt_path,
        image_width=seqinfo["width"],
        image_height=seqinfo["height"],
        gt_class_id=gt_class_id,
        min_visibility=min_visibility,
    )

    train_frames, val_frames = sample_frames(
        seq_length=seqinfo["seq_length"],
        train_count=train_count,
        val_count=val_count,
        start_frame=start_frame,
        stride=stride,
    )

    train_summary = prepare_split(
        split_name="train",
        frame_ids=train_frames,
        image_dir=image_dir,
        output_dir=output_path,
        image_ext=seqinfo["ext"],
        labels_by_frame=labels_by_frame,
        allow_empty_labels=allow_empty_labels,
    )
    val_summary = prepare_split(
        split_name="val",
        frame_ids=val_frames,
        image_dir=image_dir,
        output_dir=output_path,
        image_ext=seqinfo["ext"],
        labels_by_frame=labels_by_frame,
        allow_empty_labels=allow_empty_labels,
    )

    data_yaml_path = write_yaml(output_path)

    summary = {
        "mot_dir": str(mot_path),
        "output_dir": str(output_path),
        "sequence": seqinfo,
        "settings": {
            "train_count": train_count,
            "val_count": val_count,
            "start_frame": start_frame,
            "stride": stride,
            "gt_class_id": gt_class_id,
            "min_visibility": min_visibility,
            "allow_empty_labels": allow_empty_labels,
        },
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
        "data_yaml": str(data_yaml_path),
    }

    summary_path = output_path / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a MOT17 sequence into a small YOLO fine-tuning dataset."
    )
    parser.add_argument("--mot-dir", default="datasets/MOT17-02-SDP", help="MOT sequence directory.")
    parser.add_argument("--out", default="datasets/mot17_yolo_small", help="Output YOLO dataset directory.")
    parser.add_argument("--train-count", type=int, default=300, help="Number of train frames.")
    parser.add_argument("--val-count", type=int, default=100, help="Number of validation frames.")
    parser.add_argument("--start-frame", type=int, default=1, help="First frame id to sample.")
    parser.add_argument("--stride", type=int, default=1, help="Frame sampling stride.")
    parser.add_argument("--gt-class-id", type=int, default=1, help="MOT GT class id to keep. Pedestrian is 1.")
    parser.add_argument("--min-visibility", type=float, default=0.0, help="Minimum GT visibility.")
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Keep sampled frames even when no GT labels remain after filtering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    summary = prepare_dataset(
        mot_dir=args.mot_dir,
        output_dir=args.out,
        train_count=args.train_count,
        val_count=args.val_count,
        start_frame=args.start_frame,
        stride=args.stride,
        gt_class_id=args.gt_class_id,
        min_visibility=args.min_visibility,
        allow_empty_labels=args.allow_empty_labels,
    )

    print("\nMOT17 YOLO Dataset Prepared")
    print("===========================")
    print(f"Output:     {summary['output_dir']}")
    print(f"data.yaml:  {summary['data_yaml']}")
    for split_name, split_summary in summary["splits"].items():
        print(
            f"{split_name}: images={split_summary['images']} "
            f"labels={split_summary['label_files']} "
            f"boxes={split_summary['boxes']} "
            f"empty={split_summary['empty_label_files']}"
        )
