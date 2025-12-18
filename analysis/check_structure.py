from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root = PROJECT_ROOT / "data" / "bosch"

folders = ["train", "valid", "test"]
img_extensions = (".jpg", ".jpeg", ".png", ".bmp")


def get_files(dir_path, exts=None):
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
    img_dir = root / split / "images"
    label_dir = root / split / "labels"

    print(f"Split: {split}")
    print(f"image dir: {img_dir}")
    print(f"label dir: {label_dir}")

    if not img_dir.exists():
        print("Missing image directory")
        return

    if not label_dir.exists():
        print("Missing label directory")
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
        for name in missing_images[:10]:
            print(f"   - {name}")


def main():
    print(f"Checking Bosch dataset at: {root}")
    for split in folders:
        check_split(split)


if __name__ == "__main__":
    main()
