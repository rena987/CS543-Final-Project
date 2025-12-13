import os
from pathlib import Path

root = Path(r"C:\Users\seren\CS543-Final-Project\data\bosch")


def scan_labels_for_split(split_name: str):
    """
    Go through all label files in a split (train/valid/test) and:
    - collect unique class ids
    - count total boxes
    - flag obviously broken lines
    """
    label_dir = root / split_name / "labels"

    if not label_dir.exists():
        print(f"[{split_name}] label dir not found at: {label_dir}")
        return {
            "split": split_name,
            "unique_ids": set(),
            "num_files": 0,
            "num_empty_files": 0,
            "total_boxes": 0,
            "bad_lines": [],
        }

    unique_ids = set()
    total_boxes = 0
    num_files = 0
    num_empty_files = 0
    bad_lines = []  

    print(f"\n=== Split: {split_name} ===")
    print(f"label dir: {label_dir}")

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue

        num_files += 1
        fpath = label_dir / fname

        with open(fpath, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if not lines:
            num_empty_files += 1
            continue

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                bad_lines.append((fname, ln))
                continue

            cls_raw = parts[0]
            try:
                cls_id = int(float(cls_raw))
            except ValueError:
                bad_lines.append((fname, ln))
                continue

            unique_ids.add(cls_id)
            total_boxes += 1

    print(f"num label files: {num_files}")
    print(f"empty label files: {num_empty_files}")
    print(f"total boxes (all files): {total_boxes}")
    print(f"unique class ids in this split: {sorted(unique_ids) if unique_ids else []}")

    if bad_lines:
        print(f"bad lines found: {len(bad_lines)} (showing first 10)")
        for fname, ln in bad_lines[:10]:
            print(f"  {fname}: '{ln}'")

    return {
        "split": split_name,
        "unique_ids": unique_ids,
        "num_files": num_files,
        "num_empty_files": num_empty_files,
        "total_boxes": total_boxes,
        "bad_lines": bad_lines,
    }


def main():
    print(f"Checking Bosch labels at: {root}\n")

    all_unique_ids = set()
    splits = ["train", "valid", "test"]
    split_stats = []

    for split in splits:
        stats = scan_labels_for_split(split)
        split_stats.append(stats)
        all_unique_ids |= stats["unique_ids"]

    print("\n=== Summary across all splits ===")
    print(f"overall unique class ids: {sorted(all_unique_ids) if all_unique_ids else []}")

    if all_unique_ids:
        max_id = max(all_unique_ids)
        print(f"suggested nc (if contiguous from 0) = {max_id + 1}")
    else:
        print("no boxes found at all, which would be weird for this dataset.")


if __name__ == "__main__":
    main()
