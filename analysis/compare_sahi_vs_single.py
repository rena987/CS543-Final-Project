from pathlib import Path
import random
import csv

import cv2
import numpy as np

from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel



PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "experiments" / "bosch_sanity" / "weights" / "best.pt"
IMG_DIR = PROJECT_ROOT / "data" / "bosch" / "valid" / "images"

OUT_DIR = PROJECT_ROOT / "debug" / "compare_sahi_vs_single"
CONF = 0.25

# SAHI tiling params
SLICE_H = 512
SLICE_W = 512
OVERLAP = 0.20

# how many images to compare
NUM_IMAGES = 20
RANDOM_SEED = 0

# IoU threshold to consider "same detection"
IOU_MATCH = 0.50
IOU_DUP = 0.80


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def draw_boxes(img_bgr, boxes, color=(0, 255, 0), prefix=""):
    out = img_bgr.copy()
    for (x1, y1, x2, y2, cls_id, conf) in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix}{cls_id} {conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def get_single_shot_boxes(yolo_model, img_path):
    results = yolo_model.predict(source=str(img_path), conf=CONF, verbose=False)[0]
    boxes_out = []
    if results.boxes is None or len(results.boxes) == 0:
        return boxes_out

    xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        boxes_out.append((float(x1), float(y1), float(x2), float(y2), int(k), float(c)))
    return boxes_out


def get_sahi_boxes(sahi_model, img_path):
    result = get_sliced_prediction(
        image=str(img_path),
        detection_model=sahi_model,
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
    )

    boxes_out = []
    for op in result.object_prediction_list:
        bbox = op.bbox  # minx, miny, maxx, maxy
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        cls_id = int(op.category.id)
        conf = float(op.score.value)
        if conf >= CONF:
            boxes_out.append((float(x1), float(y1), float(x2), float(y2), cls_id, conf))
    return boxes_out


def count_duplicates(boxes):
    # count pairs that overlap a lot (rough duplicate estimate)
    dup = 0
    xyxy = [b[:4] for b in boxes]
    for i in range(len(xyxy)):
        for j in range(i + 1, len(xyxy)):
            if iou_xyxy(xyxy[i], xyxy[j]) >= IOU_DUP:
                dup += 1
    return dup


def count_new_sahi_vs_single(sahi_boxes, single_boxes):
    single_xyxy = [b[:4] for b in single_boxes]
    new_count = 0
    for sb in sahi_boxes:
        s_xyxy = sb[:4]
        matched = any(iou_xyxy(s_xyxy, bx) >= IOU_MATCH for bx in single_xyxy)
        if not matched:
            new_count += 1
    return new_count


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_single = OUT_DIR / "single"
    out_sahi = OUT_DIR / "sahi"
    out_side = OUT_DIR / "side_by_side"
    for d in [out_single, out_sahi, out_side]:
        d.mkdir(parents=True, exist_ok=True)

    # load models
    yolo_model = YOLO(str(MODEL_PATH))
    sahi_model = UltralyticsDetectionModel(
        model_path=str(MODEL_PATH),
        confidence_threshold=CONF,
        device="cpu",
    )

    # pick images
    imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    random.seed(RANDOM_SEED)
    chosen = random.sample(imgs, k=min(NUM_IMAGES, len(imgs)))

    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "num_single",
            "num_sahi",
            "num_new_sahi_vs_single",
            "dup_pairs_sahi_est",
        ])

        for img_path in chosen:
            img = cv2.imread(str(img_path))
            if img is None:
                print("Could not read:", img_path)
                continue

            single_boxes = get_single_shot_boxes(yolo_model, img_path)
            sahi_boxes = get_sahi_boxes(sahi_model, img_path)

            new_sahi = count_new_sahi_vs_single(sahi_boxes, single_boxes)
            dup_sahi = count_duplicates(sahi_boxes)

            # visuals
            vis_single = draw_boxes(img, single_boxes, color=(0, 255, 0), prefix="")
            vis_sahi = draw_boxes(img, sahi_boxes, color=(0, 0, 255), prefix="")

            # save
            out1 = out_single / img_path.name
            out2 = out_sahi / img_path.name
            cv2.imwrite(str(out1), vis_single)
            cv2.imwrite(str(out2), vis_sahi)

            # side-by-side (resize to same height)
            h = img.shape[0]
            a = cv2.resize(vis_single, (int(vis_single.shape[1] * h / vis_single.shape[0]), h))
            b = cv2.resize(vis_sahi, (int(vis_sahi.shape[1] * h / vis_sahi.shape[0]), h))
            side = np.hstack([a, b])

            # add titles
            cv2.putText(side, "Single-shot (no tiling)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(side, "SAHI tiling", (a.shape[1] + 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

            out3 = out_side / img_path.name
            cv2.imwrite(str(out3), side)

            writer.writerow([img_path.name, len(single_boxes), len(sahi_boxes), new_sahi, dup_sahi])
            print(f"{img_path.name}: single={len(single_boxes)} sahi={len(sahi_boxes)} new_sahi={new_sahi} dup_est={dup_sahi}")

    print("\nSaved:")
    print("  Side-by-side:", out_side)
    print("  CSV summary:", csv_path)


if __name__ == "__main__":
    main()
