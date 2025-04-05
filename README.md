# 🔊 Audio Deepfake Detection using Pretrained Model

This project aims to **detect AI-generated deepfake audio** using a **pretrained EfficientNet-based model** on spectrograms extracted from the **ASVspoof dataset**. The model classifies audio as either **Bonafide (real)** or **Spoof (fake)**.

---

## 📌 Table of Contents
- [🔍 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🧠 Model Architecture](#-model-architecture)
- [🚀 Methodology](#-methodology)
- [📊 Performance Metrics](#-performance-metrics)
- [🛠 Installation & Setup](#-installation--setup)
- [📂 Project Structure](#-project-structure)
- [📌 Future Improvements](#-future-improvements)
- [📜 License](#-license)
- [✉️ Contact](#-contact)

---

## 🔍 Project Overview

Deepfake audio detection is **critical** in ensuring the integrity of voice-based authentication systems, media verification, and security applications. This project focuses on:
✅ **Transforming audio data into spectrogram images**  
✅ **Using a pretrained EfficientNetB0 model** to extract meaningful patterns  
✅ **Binary classification (Bonafide vs. Spoof)**  
✅ **Evaluating model performance with accuracy, precision, recall, and F1-score**  

---

## 📊 Dataset

We use the **ASVspoof dataset**, which contains:
- **Bonafide audio** – real human speech
- **Spoofed audio** – AI-generated deepfake speech using various synthesis techniques

Each audio file is converted into a **Mel-Spectrogram** representation, which is then fed into the CNN-based model.

🔗 **Dataset Reference:** [ASVspoof Challenge](https://www.asvspoof.org/)

---

## 🧠 Model Architecture

The model is based on **EfficientNetB0**, a **lightweight CNN** optimized for high performance.

🔹 **Pretrained EfficientNetB0** extracts deep features from spectrograms  
🔹 **GlobalAveragePooling2D** reduces feature maps  
🔹 **Dense layers** with **ReLU and Dropout** prevent overfitting  
🔹 **Sigmoid activation** for binary classification  

### 🔧 Model Summary:
```plaintext
EfficientNetB0 (pretrained) → GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
```

---

## 🚀 Methodology

### 1️⃣ Data Preprocessing
- Convert raw audio files into **Mel-Spectrograms**
- Normalize and resize spectrograms to **128×128 pixels**
- Split into **train and test sets (80-20 split)**

### 2️⃣ Model Training
- **Freeze EfficientNetB0 layers** to retain pretrained knowledge
- Use **Adam optimizer (LR=0.0001)**
- Train for **10 epochs with batch size 16**

### 3️⃣ Evaluation
- Compute **Accuracy, Precision, Recall, and F1-score**
- Adjust prediction probability thresholds for optimal classification

---

## 📊 Performance Metrics

| Metric        | Value  |
|--------------|--------|
| **Accuracy**  | 71.00%  |
| **Precision** | 34.69%  |
| **Recall**    | 23.61%  |
| **F1 Score**  | 28.10%  |

🚀 The model achieves **71% accuracy**, but recall needs improvement to detect more deepfake samples correctly.

---

## 🛠 Installation & Setup

### 🔧 Prerequisites
Ensure you have Python **3.8+** installed.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AmoghAgrawal1249/Audio-Deepfake-Detection-using-Pretrained-Model.git
cd Audio-Deepfake-Detection-using-Pretrained-Model
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py
```

### 4️⃣ Evaluate the Model
```bash
python evaluate.py
```

---

## 📂 Project Structure

```
Audio-Deepfake-Detection-using-Pretrained-Model/
│── data/                  # Dataset directory (not included in repo)
│── data_loader.py         # Handles dataset loading 
│── preprocess.py          # Handles dataset preprocessing 
│── train.py               # Training script
│── evaluate.py            # Model evaluation script
│── requirements.txt       # Dependencies
│── README.md              # Documentation
```

---

## 📌 Future Improvements

🚀 **Improve Recall:** Tune model hyperparameters to detect deepfake samples more effectively.  
🎯 **Data Augmentation:** Use audio transformations to increase diversity in training data.  
📈 **Explore Transformer-Based Models:** Investigate Wav2Vec2 or AST for better feature extraction.  
⚡ **Deploy as a Web API:** Package the model into a REST API for real-world applications.  

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify!

---

## ✉️ Contact

For any questions, feel free to open an **issue** or reach out!

📧 Email: [your.email@example.com](mailto:amogh.trex@gmail.com)  
🔗 GitHub: [github.com/YOUR_USERNAME](https://github.com/AmoghAgrawal1249)  
