# Age Detection using Deep Learning in OpenCV

A two-stage computer vision pipeline built with Python and OpenCV's Deep Neural Network (`cv2.dnn`) module that detects human faces in images and classifies them into 8 discrete age brackets.

---

## Overview

This project combines two pre-trained deep learning models from different frameworks (TensorFlow and Caffe) inside OpenCV without requiring heavy external deep learning runtimes for inference:

1. **Stage 1 — Face Detection (TensorFlow):** Uses a quantized Single Shot MultiBox Detector (SSD) with a ResNet-10 backbone (`opencv_face_detector_uint8.pb`) to localize faces and extract bounding box coordinates `[x1, y1, x2, y2]`.
2. **Stage 2 — Age Classification (Caffe):** Crops each detected face with padding, normalizes it to $227 \times 227$ pixels using mean subtraction, and passes it through a pre-trained Convolutional Neural Network (`age_net.caffemodel`) designed by Gil Levi and Tal Hassner (2015) and trained on the **Adience Benchmark dataset**.

---

## Predicted Age Brackets

The Caffe classification network outputs probabilities across 8 fixed age categories:

| Index | Age Bracket |
| :---: | :---: |
| `0` | `(0-2)` |
| `1` | `(4-6)` |
| `2` | `(8-12)` |
| `3` | `(15-20)` |
| `4` | `(25-32)` |
| `5` | `(38-43)` |
| `6` | `(48-53)` |
| `7` | `(60-100)` |

---

## Project Structure

```text
DL AGE_DETACTION/
├── Age_Detection_using_Deep_Learning_in_OpenCV.ipynb
├── model_dir/
│   ├── age_deploy.prototxt              # Caffe CNN architecture configuration
│   ├── age_net.caffemodel               # Pre-trained Caffe age classification weights
│   ├── opencv_face_detector.pbtxt       # TensorFlow SSD-ResNet10 graph configuration
│   └── opencv_face_detector_uint8.pb    # Quantized TensorFlow face detection weights
└── sample_photos/
    ├── 50_year_old women.jpg
    ├── 8_10_old.jpg
    ├── baby.jpg
    ├── kid_girl.jpg
    ├── mahant_swami.jpg
    ├── mr_beast.jpg
    └── teenager.jpg

## Tech Stack

* **Language:** Python 3
* **Computer Vision & Inference:** OpenCV (`cv2`, `cv2.dnn`)
* **Numerical & Visualization Tools:** NumPy, Matplotlib
* **Environment:** Jupyter Notebook
* **Pre-trained Models:**
  * TensorFlow SSD-ResNet10 Face Detector
  * Caffe AgeNet (Levi & Hassner, CVPR 2015)

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/DakshMepani001/Age-Detection-OpenCV.git](https://github.com/DakshMepani001/Age-Detection-OpenCV.git)
   cd Age-Detection-OpenCV
   ```

2. **Install required dependencies:**
   ```bash
   pip install opencv-python numpy matplotlib jupyter
   ```

3. **Verify model files:**
   Ensure all four model files (`age_deploy.prototxt`, `age_net.caffemodel`, `opencv_face_detector.pbtxt`, `opencv_face_detector_uint8.pb`) are present inside the `model_dir/` directory.

---

## Usage

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. Open `Age_Detection_using_Deep_Learning_in_OpenCV.ipynb`.

3. Update `image_path` to point to any image inside `sample_photos/` (or your own custom image):
   ```python
   image_path = r"sample_photos/kid_girl.jpg"
   process_image(image_path)
   ```

4. Run all cells to view the detected bounding boxes and predicted age brackets rendered inline.

---

## Known Limitations

* **Dataset Bias:** Because the Adience dataset is heavily skewed toward young adults, the model can occasionally bias uncertain middle-aged predictions toward the `(25-32)` bracket.
* **Resolution Constraints:** Input face crops are resized to `227 x 227`, which can smooth out fine skin textures and subtle age markers under soft or bright lighting.
* **Fixed Output Classes:** The 8 age ranges are hardcoded into the final Softmax layer of `age_net.caffemodel` and cannot be modified without retraining the network.

---

## References

* Gil Levi and Tal Hassner, *"Age and Gender Classification using Convolutional Neural Networks"*, IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), CVPR 2015.
* OpenCV Deep Neural Network (`cv2.dnn`) Module Documentation.
