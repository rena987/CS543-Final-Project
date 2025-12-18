from pathlib import Path
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# Paths
MODEL_PATH = Path("experiments/bosch_sanity/weights/best.pt")
IMG_DIR = Path("data/bosch/valid/images")
OUT_DIR = Path("debug/sahi_tiling_valid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

detection_model = UltralyticsDetectionModel(
    model_path=str(MODEL_PATH),
    confidence_threshold=0.25,
    device="cpu",  
)

SLICE_H, SLICE_W = 640, 640
OVERLAP_H, OVERLAP_W = 0.2, 0.2
MAX_IMAGES = 50

imgs = sorted([p for p in IMG_DIR.glob("*.jpg")])
if MAX_IMAGES:
    imgs = imgs[:MAX_IMAGES]

print(f"Running SAHI on {len(imgs)} images...")
for i, img_path in enumerate(imgs, 1):
    result = get_sliced_prediction(
        image=str(img_path),
        detection_model=detection_model,
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP_H,
        overlap_width_ratio=OVERLAP_W,
    )
    result.export_visuals(export_dir=str(OUT_DIR))
    if i % 10 == 0:
        print(f"  done {i}/{len(imgs)}")

print("Saved SAHI visuals to:", OUT_DIR)
