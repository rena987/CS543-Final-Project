# CS543-Final-Project

Traffic light detection on the **Bosch Small Traffic Lights** dataset using **YOLOv8** (so far, need to add SAHI Tiling and other augmentations as we go on)

---

## Dataset

- Dataset: **Bosch Small Traffic Lights** 
- Source: Public Bosch traffic lights project on **Roboflow** (exported as YOLOv8 from https://universe.roboflow.com/nwpuvhr10-dwb8w/bosch-small-traffic-lights/browse/fork?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- Local root: C:{personal_path}\CS543-Final-Project\data\bosch
- Final folder structure
    data/
    bosch/
        data.yaml
        README.dataset.txt
        README.roboflow.txt
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
- data.yaml format:
    path: C:{personal_path}\CS543-Final-Project\data\bosch
    train: train/images
    val: valid/images
    test: test/images
    nc: 5
    names: [class0, class1, class2, class3, class4]

## YOLO Experiments

All YOLO commands are run from:

    cd C:{personal_path}\CS543-Final-Project

and log outputs into the `experiments/` directory.

---

### 1. One-epoch sanity run (`bosch_sanity_v1`)

**Purpose**  
Quick check that the dataset and labels are wired correctly, the model runs end-to-end, and the outputs (plots and predictions) look sane.

**Command (PowerShell):**

    yolo detect train `
      data="C:{personal_path}\CS543-Final-Project\data\bosch\data.yaml" `
      model=yolov8n.pt `
      imgsz=640 `
      epochs=1 `
      batch=8 `
      workers=0 `
      project=experiments `
      name=bosch_sanity_v1 `
      plots=True

**Key outputs (under `experiments/bosch_sanity_v1/`):**

- `weights/best.pt` – checkpoint after 1 epoch  
- `results.csv` – loss and metric values  

Sanity plots:

- `labels.jpg` – class distribution + box size & location stats  
- `confusion_matrix_normalized.png`  
- `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png`  

Example batches:

- `val_batch*_labels.jpg` – ground truth boxes  
- `val_batch*_pred.jpg` – model predictions after 1 epoch  

These are used to visually confirm:

- class imbalance (e.g., class 2 very frequent, class 4 rare)  
- bounding boxes live in realistic locations/sizes  
- the model starts to learn something even after a single epoch.

---

### 2. Alternative sanity run (`bosch_sanity_run3`)

**Purpose**  
Same idea as above but with slightly different CLI syntax and batch size, mainly to test runtime and configuration.

**Command:**

    yolo task=detect mode=train `
      model=yolov8n.pt `
      data="C:{personal_path}\CS543-Final-Project\data\bosch\data.yaml" `
      epochs=1 `
      imgsz=640 `
      batch=16 `
      workers=0 `
      project=experiments `
      name=bosch_sanity_run3 `
      plots=False

**Outputs (under `experiments/bosch_sanity_run3/`):**

- `weights/best.pt`  
- `results.csv`  

No plots are saved here (`plots=False`); this run is mainly a quick timing + config sanity check.

---

### 3. Main CPU baseline (`bosch_yolov8n_fast_v1`)

**Purpose**  
Train a longer YOLOv8n baseline on CPU only, but keep it small enough to be feasible without a GPU.  
This is the main Bosch YOLO baseline that I will analyze in my report.

**Command:**

    yolo detect train `
      data="C:{personal_path}\CS543-Final-Project\data\bosch\data.yaml" `
      model=yolov8n.pt `
      imgsz=480 `
      epochs=10 `
      batch=8 `
      workers=0 `
      project=experiments `
      name=bosch_yolov8n_fast_v1 `
      plots=False

**Notes:**

- `imgsz=480` and `batch=8` are chosen to make CPU training faster than the default 640.  
- `workers=0` avoids Windows dataloader issues.  
- `plots=False` reduces overhead during training; plots can be regenerated later if needed.

**Outputs (under `experiments/bosch_yolov8n_fast_v1/`):**

- `weights/best.pt` – main Bosch YOLO baseline checkpoint  
- `weights/last.pt`  
- `results.csv` – per-epoch losses and metrics  

This run serves as the YOLOv8 baseline for Bosch traffic light detection, and all later analysis (mAP, PR/F1 curves, confusion matrix, etc.) will refer back to this experiment.

