from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATHS = [
    PROJECT_ROOT / "debug" / "sahi_ablation" / "tile640_ov10" / "summary.csv",
    PROJECT_ROOT / "debug" / "sahi_ablation" / "tile640_ov20" / "summary.csv",
    PROJECT_ROOT / "debug" / "sahi_ablation" / "tile640_ov30" / "summary.csv",
]

OUT_DIR = PROJECT_ROOT / "debug" / "sahi_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image": r["image"],
                "num_single": int(r["num_single"]),
                "num_sahi": int(r["num_sahi"]),
                "num_new": int(r["num_new_sahi_vs_single"]),
                "dup_pairs": int(r["dup_pairs_sahi_est"]),
            })
    return rows

def summarize(rows):
    n = len(rows)
    total_single = sum(r["num_single"] for r in rows)
    total_sahi = sum(r["num_sahi"] for r in rows)
    total_new = sum(r["num_new"] for r in rows)
    total_dup_pairs = sum(r["dup_pairs"] for r in rows)

    imgs_single_ge1 = sum(1 for r in rows if r["num_single"] > 0)
    imgs_sahi_ge1 = sum(1 for r in rows if r["num_sahi"] > 0)
    imgs_new_ge1 = sum(1 for r in rows if r["num_new"] > 0)
    imgs_dup_ge1 = sum(1 for r in rows if r["dup_pairs"] > 0)

    return {
        "images": n,
        "total_single": total_single,
        "total_sahi": total_sahi,
        "avg_single": total_single / n if n else 0.0,
        "avg_sahi": total_sahi / n if n else 0.0,
        "imgs_single_ge1": imgs_single_ge1,
        "imgs_sahi_ge1": imgs_sahi_ge1,
        "imgs_new_ge1": imgs_new_ge1,
        "total_new": total_new,
        "imgs_dup_ge1": imgs_dup_ge1,
        "total_dup_pairs": total_dup_pairs,
    }

def main():
    all_summaries = []

    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"[MISSING] {csv_path}")
            continue

        rows = load_rows(csv_path)
        s = summarize(rows)

        label = csv_path.parent.name
        s["run"] = label
        all_summaries.append(s)

    out_csv = OUT_DIR / "ablation_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run","images",
            "total_single","total_sahi","avg_single","avg_sahi",
            "imgs_single_ge1","imgs_sahi_ge1",
            "imgs_new_ge1","total_new",
            "imgs_dup_ge1","total_dup_pairs"
        ])
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(s)

    print("\n=== SAHI Ablation Summary (copy into report) ===\n")
    print("| Run | Images | Total single | Total SAHI | Avg single | Avg SAHI | Img single≥1 | Img SAHI≥1 | Img new≥1 | Total new | Img dup≥1 | Dup pairs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in all_summaries:
        print(
            f"| {s['run']} | {s['images']} | {s['total_single']} | {s['total_sahi']} | "
            f"{s['avg_single']:.2f} | {s['avg_sahi']:.2f} | "
            f"{s['imgs_single_ge1']} | {s['imgs_sahi_ge1']} | "
            f"{s['imgs_new_ge1']} | {s['total_new']} | "
            f"{s['imgs_dup_ge1']} | {s['total_dup_pairs']} |"
        )

    print(f"\nSaved combined summary to: {out_csv}")

if __name__ == "__main__":
    main()
