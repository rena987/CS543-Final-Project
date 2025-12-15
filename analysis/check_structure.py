"""
Quick sanity check for Bosch dataset file pairing.

Checks that for each split (train/valid/test):
- every image in images/ has a matching .txt file in labels/
- every label in labels/ has a matching image in images/
"""

import os
from pathlib import Path
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root = PROJECT_ROOT / "data" / "bosch"


#root = Path(r"C:\Users\seren\CS543-Final-Project\data\bosch")
folders = ["train", "valid", "test"]
img_extensions = (".jpg", ".jpeg", ".png", ".bmp")


def get_files(dir_path, exts=None):
    """Return list of files in dir_path filtered by extension (if given)."""
    if not dir_path.exists():
        return []

    if exts is None:
        return [p for p in dir_path.iterdir() if p.is_file()]

    files = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def check_split(split):
    """Check one split: train / valid / test."""
    img_dir = root / split / "images"
    label_dir = root / split / "labels"

    print(f"\n=== Split: {split} ===")
    print(f"image dir: {img_dir}")
    print(f"label dir: {label_dir}")

    if not img_dir.exists():
        print("MISSING image directory, skipping")
        return

    if not label_dir.exists():
        print("MISSING label directory, skipping")
        return

    images = get_files(img_dir, img_extensions)
    labels = get_files(label_dir, (".txt",))

    img_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(img_stems - label_stems)
    missing_images = sorted(label_stems - img_stems)

    print(f"num images: {len(images)}")
    print(f"num labels: {len(labels)}")

    print(f"images with no label: {len(missing_labels)}")
    if missing_labels:
        print("  (showing up to 10)")
        for name in missing_labels[:10]:
            print(f"   - {name}")

    print(f"labels with no image: {len(missing_images)}")
    if missing_images:
        print("  (showing up to 10)")
        for name in missing_images[:10]:
            print(f"   - {name}")


def main():
    print(f"Checking Bosch dataset at: {root}")
    for split in folders:
        check_split(split)


if __name__ == "__main__":
    main()
