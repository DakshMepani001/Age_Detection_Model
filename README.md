# 🎯 Age Detection Using Deep Learning & OpenCV

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</p>

<p align="center">
  **A Deep Learning-based Computer Vision Project for Face Detection and Age Group Classification using Python and OpenCV.**
</p>

---

## 📌 Project Overview

This project implements a **two-stage Deep Learning pipeline** using Python and OpenCV's Deep Neural Network (`cv2.dnn`) module to detect human faces and estimate their age groups from images.

It integrates two pre-trained deep learning models developed using different frameworks (TensorFlow and Caffe), allowing both models to run through OpenCV without requiring separate TensorFlow or Caffe inference environments.

### ⚙️ How It Works

The complete pipeline consists of two stages:

**1️⃣ Face Detection — TensorFlow SSD + ResNet-10**

- Uses a pre-trained, quantized Single Shot MultiBox Detector (SSD) with a ResNet-10 backbone.
- Detects human faces in the input image.
- Generates bounding box coordinates: `[x1, y1, x2, y2]`.
- Model: `opencv_face_detector_uint8.pb`

**2️⃣ Age Classification — Caffe CNN**

- Extracts the detected face region with additional padding.
- Resizes the face image to `227 × 227` pixels.
- Applies mean subtraction for input normalization.
- Passes the processed face through a pre-trained CNN.
- Predicts one of eight age brackets using Softmax classification.

The age classification model was developed by **Gil Levi and Tal Hassner (2015)** and trained on the Adience Benchmark dataset.

---

## 🧠 Age Classification Categories

The model predicts probabilities across eight predefined age groups.

| Index | Predicted Age Group |
|:---:|:---:|
| `0` | 👶 0–2 years |
| `1` | 🧒 4–6 years |
| `2` | 👦 8–12 years |
| `3` | 🧑 15–20 years |
| `4` | 👨 25–32 years |
| `5` | 👨‍🦱 38–43 years |
| `6` | 👨‍🦳 48–53 years |
| `7` | 👴 60–100 years |

> **Note:** These are predefined age categories, not exact age predictions. The model classifies faces into the closest learned age bracket.

---

## 🛠️ Technology Stack

<p align="left">
  <img src="https://skillicons.dev/icons?i=python,opencv,tensorflow,numpy,matplotlib,jupyter" />
</p>

| Category | Technologies |
|---|---|
| 💻 Programming Language | Python 3 |
| 👁️ Computer Vision | OpenCV |
| 🧠 Deep Learning | CNN, SSD, ResNet-10 |
| ⚡ Inference Engine | OpenCV DNN (`cv2.dnn`) |
| 🔢 Numerical Computing | NumPy |
| 📊 Visualization | Matplotlib |
| 📓 Development Environment | Jupyter Notebook |
| 📦 Model Formats | TensorFlow `.pb`, Caffe `.caffemodel` |

### Pre-trained Models

- **Face Detection:** TensorFlow SSD-ResNet10
- **Age Classification:** Caffe AgeNet (Levi & Hassner, 2015)

---

## 📂 Project Structure

```text
Age-Detection-OpenCV/
│
├── 📓 Age_Detection_using_Deep_Learning_in_OpenCV.ipynb
│
├── 📁 model_dir/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
│
├── 📁 sample_photos/
│   ├── 50_year_old women.jpg
│   ├── 8_10_old.jpg
│   ├── baby.jpg
│   ├── kid_girl.jpg
│   ├── mahant_swami.jpg
│   ├── mr_beast.jpg
│   └── teenager.jpg
│
└── 📄 README.md
```

---

## 🚀 Installation & Setup

Follow these steps to run the project on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DakshMepani001/Age-Detection-OpenCV.git

cd Age-Detection-OpenCV
```

### 2️⃣ Install Dependencies

Make sure Python 3 is installed, then execute:

```bash
pip install opencv-python numpy matplotlib jupyter
```

### 3️⃣ Verify Pre-trained Models

Ensure that all four model files are available inside the `model_dir/` directory:

- `age_deploy.prototxt`
- `age_net.caffemodel`
- `opencv_face_detector.pbtxt`
- `opencv_face_detector_uint8.pb`

These files are essential for face detection and age classification.

---

## ▶️ Usage

### 1. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 2. Open the Project Notebook

Open:

```text
Age_Detection_using_Deep_Learning_in_OpenCV.ipynb
```

### 3. Select an Input Image

Update the `image_path` variable with the path of your desired image.

```python
image_path = r"sample_photos/kid_girl.jpg"

process_image(image_path)
```

You can also provide your own custom image.

### 4. Execute the Notebook

Run all notebook cells to:

- Detect faces in the input image.
- Extract detected face regions.
- Perform age group classification.
- Display the output image with bounding boxes and predicted age labels.

---

## ⚠️ Known Limitations

Although the model provides a useful demonstration of deep learning-based age estimation, it has some limitations.

| Limitation | Description |
|---|---|
| 📊 Dataset Bias | The Adience dataset has demographic and age-distribution limitations that can affect prediction accuracy. |
| 🖼️ Image Resolution | Resizing face crops to `227 × 227` may remove fine-grained facial details. |
| 🎯 Fixed Age Categories | The model predicts only eight predefined age brackets. |
| 💡 Lighting Conditions | Poor lighting, shadows, and overexposure may affect face detection and classification. |
| 📐 Face Orientation | Extreme head angles, occlusions, and partially visible faces may reduce detection performance. |

**Important:** Age predictions are approximate and should not be treated as verified or exact ages.

---

## 📚 References & Acknowledgements

1. **Research Paper:** Gil Levi and Tal Hassner, *Age and Gender Classification using Convolutional Neural Networks*, IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), CVPR 2015.
2. **OpenCV Documentation:** [OpenCV Deep Neural Network Module](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
3. **Dataset:** [Adience Benchmark Dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html)

---

## 👨‍💻 Author

**Daksh Mepani**

🎓 Interested in **Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision**.

- 🐙 GitHub: [@DakshMepani001](https://github.com/DakshMepani001)

---

<p align="center">
  ⭐ If you find this project interesting, consider giving it a star!
</p>

<p align="center">
  **Built with ❤️ using Python, Deep Learning & OpenCV**
</p>
