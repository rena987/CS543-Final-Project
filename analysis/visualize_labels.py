import os
import random
from pathlib import Path
import cv2

#root = Path(r"C:\Users\seren\CS543-Final-Project\data\bosch")
#output = Path(r"C:\Users\seren\CS543-Final-Project\debug\visualize_labels")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root = PROJECT_ROOT / "data" / "bosch"
output = PROJECT_ROOT / "debug" / "visualize_labels"

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
root = PROJECT_ROOT / "data" / "bosch"

output = PROJECT_ROOT / "debug" / "visualize_labels"


def draw_boxes_for_split(split_name: str, num_images: int = 5):
    """
    For a given split (train/valid/test), randomly pick a few images,
    draw YOLO boxes on them, and save to output.

    This is just to visually sanity-check that:
    - boxes align with traffic lights
    - coords are not flipped or off-image
    """
    img_dir = root / split_name / "images"
    label_dir = root / split_name / "labels"

    if not img_dir.exists() or not label_dir.exists():
        print(f"[{split_name}] missing image or label dir")
        print(f"  img_dir:   {img_dir}")
        print(f"  label_dir: {label_dir}")
        return

    print(f"\n=== Visualizing split: {split_name} ===")
    print(f"image dir: {img_dir}")
    print(f"label dir: {label_dir}")

    img_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not img_files:
        print(f"[{split_name}] no image files found.")
        return

    sample_files = random.sample(
        img_files,
        k=min(num_images, len(img_files))
    )

    split_out_dir = output / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    for fname in sample_files:
        img_path = img_dir / fname
        stem = os.path.splitext(fname)[0]
        label_path = label_dir / f"{stem}.txt"

        if not label_path.exists():
            print(f"  skipping {fname}: no label file")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  could not read image: {img_path}")
            continue

        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                print(f"  bad line in {label_path.name}: '{ln}'")
                continue

            cls_id = int(float(parts[0]))
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{cls_id}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        out_path = split_out_dir / f"{stem}_debug.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"  saved: {out_path}")


def main():
    print(f"Saving visualizations to: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        draw_boxes_for_split(split_name=split, num_images=5)


if __name__ == "__main__":
    main()
