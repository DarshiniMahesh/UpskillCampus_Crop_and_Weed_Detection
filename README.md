# Crop & Weed Detection

A computer vision / deep learning project to detect and differentiate between crops and weeds in agricultural fields. This tool aims to help automate weed control, reduce herbicide usage, and support precision farming.

## Overview

Agricultural fields are often invaded by weeds, which reduce crop yield and increase maintenance cost. This project uses object detection methods to identify crops and weeds in images captured from fields, enabling automated or semi-automated weed removal.  

The repository includes data processing, training, inference scripts, and a simple web app interface (via Streamlit) for demonstration.

## Features

- Trainable object detection model (e.g. using YOLO or similar)  
- Inference on new images to highlight weeds vs crops  
- Web UI for quick testing  
- Modular scripts to adapt to different datasets or models  

## Project Structure

```

.
├── data/                     # raw and processed datasets
├── models/                   # saved model weights & architectures
├── outputs/                  # inference results, predictions, visualizations
├── scripts/                  # helper scripts (e.g. preprocessing, utilities)
├── streamlit_app.py          # front-end web app
├── train_and_infer.py        # training & inference orchestration
├── requirements.txt          # Python dependencies
└── yolov5su.pt               # pretrained or baseline model weights

````

- **data/** — Contains your training, validation, and test image sets and annotation files.  
- **models/** — Stores trained model weights, checkpoints, model definitions.  
- **outputs/** — Where the generated prediction images, logs, and metrics are saved.  
- **scripts/** — Utility scripts (e.g. annotation parsing, image augmentations).  
- **streamlit_app.py** — Runs a web front-end to upload images and display predictions.  
- **train_and_infer.py** — Core script to train the model or run inference.  
- **requirements.txt** — Lists all necessary Python packages.  
- **yolov5su.pt** — A default / baseline model weight file included (you may replace or retrain it).

## Getting Started

### Prerequisites

- Python 3.7+  
- GPU (CUDA-enabled) is recommended for training / faster inference  
- pip (or conda) to install packages  

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DeepakMallesh/upskillcampus_Crop-and-Weed-Detection.git
   cd upskillcampus_Crop-and-Weed-Detection
````

2. Create & activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

There are two main modes:

* **Training / inference via CLI**
* **Web interface via Streamlit**

#### Command-Line Usage (train & inference)

```bash
python train_and_infer.py --mode train --config path/to/config.yaml
python train_and_infer.py --mode infer --input path/to/images --output outputs/
```

You can customize config files (if supported) to set hyperparameters, augmentation, paths, etc.

#### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

Then open the provided local URL (e.g. `http://localhost:8501`) in your browser. You’ll see a simple interface to upload an image and get predictions.

## Model Training & Inference

* Training uses standard object detection pipelines (e.g. YOLO variants)
* Supports augmentation, learning rate schedule, checkpointing
* Inference produces bounding boxes distinguishing between *crop* and *weed* classes
* Predictions saved as annotated images in **outputs/** along with CSV/JSON of detections

## Datasets

COCO or YOLO format

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Ensure that classes are properly labeled (e.g. `0 = crop`, `1 = weed`, etc.).

## Results & Outputs

After inference, annotated images showing detected weeds vs crops are saved under **outputs/**. You may also produce metrics like mAP, precision, recall, confusion matrix, etc.

You can compare baseline model **yolov5su.pt** performance against your retrained weights in **models/**.

## Contributing

Contributions, ideas, improvements are welcome! Here’s how you can help:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes / enhancements
4. Ensure code is clean, documented, and tested
5. Open a Pull Request

Please adhere to best practices and include explanations / comments in your PR.

## Contact

If you have questions or want to collaborate:

* Author: Darshini Mahesh
* GitHub: [DarshiniMahesh](https://github.com/DarshiniMahesh)
* Email: [darshinims00@gmail.com](mailto:darshinims00@gmail.com)
