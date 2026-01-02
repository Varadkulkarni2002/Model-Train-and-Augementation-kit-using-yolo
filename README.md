# YOLO Training Pipeline Toolkit

This repository provides a **unified set of tools for preparing datasets and training YOLO models**.  
It bundles dataset augmentation, dataset YAML configuration generation, and a GUI-based YOLOv8 training dashboard into a single, structured workflow.

The focus of this repository is **practical training pipelines**, not experimentation scripts or auto-labeling tools.

---

## What This Repository Contains

This repository groups tools that are **used together during the model training phase** of a Computer Vision project.

### 1. Dataset Augmentation Tool
Used to expand and balance datasets before training.

**Purpose**
- Increase dataset size
- Improve class balance
- Improve model generalization

---

### 2. YOLO Dataset YAML Generator
Generates a valid YOLOv8-compatible `data.yaml` file.

**Purpose**
- Convert dataset folders into training-ready configuration
- Auto-generate `train`, `val`, and `test` paths
- Handle dataset splits directly from the GUI

---

### 3. YOLOv8 Training Dashboard
A desktop GUI for training and exporting YOLOv8 models.

**Purpose**
- Configure training parameters
- Start and monitor training runs
- Export trained models (ONNX / TorchScript)

---

## What This Repository Is NOT

- ❌ Not an annotation tool
- ❌ Not an auto-labeling system
- ❌ Not an inference or deployment pipeline

This separation is intentional.

---

## Installation

### Requirements
- Python 3.8+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:**  
> `tkinter` is bundled with most Python installations.  
> On Linux systems:
```bash
sudo apt install python3-tk
```

---

## Repository Structure

```
yolo-training-pipeline/
├── data_augmentation/
│   └── augmentor.py
├── dataset_yaml_generator/
│   └── yaml_generator.py
├── model_training/
│   └── model_trainer.py
├── requirements.txt
└── README.md
```

---

## Usage Workflow (Recommended Order)

### 1️⃣ Dataset Augmentation
Run the augmentation tool to expand or balance your dataset.

```bash
python data_augmentation/augmentor.py
```

---

### 2️⃣ Generate YOLO Dataset YAML
Create a YOLO-compatible `data.yaml` file.

```bash
python dataset_yaml_generator/yaml_generator.py
```

This step can also:
- Auto-split datasets (train / val / test)
- Preview YAML before saving

---

### 3️⃣ Train YOLO Model
Launch the training dashboard.

```bash
python model_training/model_trainer.py
```

From the GUI you can:
- Select model size (YOLOv8n/s/m/l/x)
- Configure epochs, batch size, image size
- Choose CPU or GPU
- Export trained models

---

## Typical Use Cases

- Training YOLO models on custom datasets
- Rapid experimentation with dataset variations
- Academic and industrial computer vision projects
- End-to-end training pipelines without CLI complexity

---

## Limitations

- Does not include data annotation or inspection
- Designed for single-user, local workflows
- Assumes YOLOv8-compatible datasets

---

## License

MIT License

---

## Author

Varad Kulkarni  
Applied AI / Computer Vision
