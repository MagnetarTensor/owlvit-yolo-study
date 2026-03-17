# YOLOv8 vs OWL-ViT — Comparative Object Detection Study

A comparative study of two object detection paradigms on a custom COCO-format dataset:
- **YOLOv8** — supervised detection, trained on COCO
- **OWL-ViT** — zero-shot detection via Vision-Language Model (Google)

The study evaluates both models on **seen** and **unseen** classes, benchmarks inference latency, and explores prompt engineering strategies for zero-shot detection.

---

## Key findings

### Detection performance

| Model | mAP@[.5:.95] (seen) | AP@50 (seen) | mAP@[.5:.95] (unseen) | AP@50 (unseen) | Latency (s/img) |
|---|---|---|---|---|---|
| YOLOv8n | 0.549 | 0.645 | 0.000 | 0.000 | 0.78 |
| OWL-ViT | 0.600 | 0.718 | 0.331 | 0.522 | 3.56 |

Dataset: 246 images, 453 annotations, 10 categories (5 seen / 5 unseen).

**Key takeaways:**
- OWL-ViT slightly outperforms YOLOv8 on seen classes (mAP +0.05) while being ~4.6× slower
- YOLOv8 scores 0 on unseen classes — expected, as it cannot generalize beyond its training categories
- OWL-ViT achieves mAP 0.33 on unseen classes via zero-shot text prompts, demonstrating strong generalization

### Per-class breakdown (seen classes)

| Class | OWL-ViT AP@50 | YOLOv8 AP@50 |
|---|---|---|
| Pizza | 0.971 | 0.858 |
| Dog | 0.957 | 0.875 |
| Bus | 0.856 | 0.840 |
| Bicycle | 0.661 | 0.513 |
| Chair | 0.145 | 0.140 |

### Prompt engineering — Helmet (unseen class)

| Prompt | mAP@[.5:.95] | AP@50 |
|---|---|---|
| "a photo of a helmet" | 0.133 | 0.194 |
| "a black helmet" | 0.095 | 0.126 |
| "a motorcycle helmet" | 0.030 | 0.064 |
| "a construction helmet" | 0.014 | 0.020 |
| "a person wearing a helmet" | 0.002 | 0.007 |
| "a safety helmet" | 0.000 | 0.000 |

Generic prompts outperform specific ones — the model was likely trained on generic image-text pairs, making overly specific prompts harder to match.

### Prompt specificity experiment — Red bus

Demonstrates that adding a color attribute to the prompt ("a photo of a red bus") enables OWL-ViT to detect an object it missed with a generic prompt — score 0.50 vs no detection.

---

## What's inside

### Models compared

**YOLOv8** — single-pass detector trained on COCO. Fast, accurate on seen classes, but requires retraining for new categories.

**OWL-ViT** — Vision-Language Model from Google. Takes text prompts as input ("a photo of a helmet") and detects matching objects zero-shot, without any retraining.

### Evaluation protocol

- **Seen classes** — classes present in both the training data and the evaluation set (Bicycle, Dog, Pizza, Bus, Chair)
- **Unseen classes** — classes never seen during training, detected only via text prompts (Helmet, ...)
- Metrics: **mAP@[0.5:0.95]**, **AP50**, **AP75**, **AR@100** using pycocotools COCOeval

### Prompt engineering

Explores how prompt formulation affects OWL-ViT's zero-shot performance on unseen classes:
```python
PROMPTS = [
    "a photo of a helmet",
    "a black helmet",
    "a motorcycle helmet",
    "a person wearing a helmet",
    "a safety helmet",
    "a construction helmet",
]
```
Each prompt is evaluated independently — AP score per prompt, ranked.

### Timing benchmark

30-image benchmark comparing inference speed between YOLOv8 and OWL-ViT on CPU and GPU.

---

## Installation

```bash
git clone https://github.com/MagnetarTensor/owlvit-yolo-study.git
cd owlvit-yolo-study
pip install ultralytics transformers accelerate pycocotools
```

---

## Dataset structure

The full dataset images are not included in this repo (too large). A set of sample images is available in `data/samples/` to run the notebook on a small subset.

To run on the full dataset, place your images in `data/images/`:

```
data/
├── images/          ← full dataset images (not tracked by git)
├── samples/         ← sample images included in repo
├── annotations.json ← COCO-format annotations
└── INFO.txt         ← dataset info
```

The annotations file follows the COCO format:
```json
{
  "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category": "Dog", "bbox": [x, y, w, h]}],
  "info": {}
}
```

To run on samples only, update the path in the Parameters cell:
```python
IMG_DIR = DATA_DIR / "samples"
```

---

## Usage

Open `owlvit-yolo-study.ipynb` in Jupyter or VS Code and run cells sequentially.

The notebook is structured as follows:

| Section | Description |
|---|---|
| Loading & Imports | Dependencies and data loading |
| Parameters | Data paths and config |
| Ground Truth Visualisation | Visualize GT bounding boxes |
| YOLOv8 Detection | Run and visualize YOLO predictions |
| OWL-ViT Detection | Run and visualize OWL-ViT predictions |
| COCO Evaluation | mAP/AP50/AP75/AR metrics for both models |
| Results Table | Side-by-side comparison |
| Timing Benchmark | Latency comparison on 30 images |
| Seen / Unseen / Hard splits | Image selection for targeted evaluation |
| GT vs YOLO vs OWL overlay | 3-way visual comparison |
| Prompt Engineering | Zero-shot performance per prompt on unseen classes |

---

## Technical notes

- OWL-ViT uses **contrastive learning** (similar to CLIP) to align image and text representations — this enables zero-shot detection without any class-specific training
- YOLOv8 processes the image in a **single forward pass** through the network — fast but limited to trained classes
- The mAP gap between seen and unseen classes quantifies the **zero-shot generalization** capability of OWL-ViT vs the supervised ceiling of YOLOv8
- Prompt engineering results show that specificity ("a motorcycle helmet") often outperforms generic prompts ("a photo of a helmet") — consistent with CLIP-style models' sensitivity to prompt formulation

---

## Author

Benjamin Madar — [LinkedIn](https://www.linkedin.com/in/benjamin-madar) | [GitHub](https://github.com/MagnetarTensor)