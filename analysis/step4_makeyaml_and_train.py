from pathlib import Path
from ultralytics import YOLO

def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "experiments").exists():
            return parent
    return Path(__file__).resolve().parents[1]

def main():
    project_root = find_project_root()

    data_yaml = project_root / "data" / "bosch" / "data.yaml"
    assert data_yaml.exists(), f"Missing dataset yaml: {data_yaml}"

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        imgsz=1280,
        epochs=50,
        batch=8,          
        patience=10,
        close_mosaic=10,
        project=str(project_root / "experiments"),
        name="step4_aug_imgsz1280",
        pretrained=True,
        device=0,         
    )

if __name__ == "__main__":
    main()
