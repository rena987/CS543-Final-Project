import os
import random
from pathlib import Path
from collections import Counter
import cv2


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "bosch"
VIS_ROOT = ROOT / "analysis_outputs" / "bosch_vis"


def draw_yolo_boxes(img, label_path, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]

    if not label_path.exists():
        return img

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x_c, y_c, bw, bh = parts
            x_c = float(x_c) * w
            y_c = float(y_c) * h
            bw = float(bw) * w
            bh = float(bh) * h

            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                img,
                "traffic_light",
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    return img


def summarize_split(split_name, num_vis=15):
    img_dir = DATA_ROOT / split_name / "images"
    lbl_dir = DATA_ROOT / split_name / "labels"

    if not img_dir.exists():
        print(f"[{split_name}] image directory not found at {img_dir}, skipping.")
        return

    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    num_images = len(image_files)

    num_boxes = 0
    class_counts = Counter()

    for img_path in image_files:
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with label_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                num_boxes += 1
                class_counts[cls_id] += 1

    print(f"Split: {split_name}")
    print(f"# of images: {num_images}")
    print(f"# of boxes:  {num_boxes}")
    print(f"class_counts: {dict(class_counts)}")
    if num_images > 0:
        print(f"average boxes per image: {num_boxes / num_images:.3f}")
    print()

    out_dir = VIS_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    vis_files = image_files.copy()
    random.shuffle(vis_files)
    vis_files = vis_files[:num_vis]

    for img_path in vis_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        label_path = lbl_dir / (img_path.stem + ".txt")
        img = draw_yolo_boxes(img, label_path)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)

    print(f"[{split_name}] Saved {len(vis_files)} overlay examples to {out_dir}")
    print()


def main():
    print(f"Project root: {ROOT}")
    print(f"Data root:    {DATA_ROOT}")
    print()

    for split in ["train", "valid", "test"]:
        summarize_split(split_name=split, num_vis=15)


if __name__ == "__main__":
    main()
