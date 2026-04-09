# 🩺 Chest X-Ray Pneumonia Classification using CNN

## 📌 Overview

This project focuses on developing a **Convolutional Neural Network (CNN)** model to automatically classify chest X-ray images into two categories:

* ✅ **NORMAL**
* ⚠️ **PNEUMONIA**

The dataset consists of **5,863 pediatric chest X-ray images**, collected from routine clinical care. The goal is to assist in medical diagnosis by building an intelligent image classification system.

---

## 📂 Dataset Description

The dataset is organized into three main folders:

```
dataset/
 ├── train/
 │    ├── NORMAL/
 │    └── PNEUMONIA/
 ├── val/
 │    ├── NORMAL/
 │    └── PNEUMONIA/
 └── test/
      ├── NORMAL/
      └── PNEUMONIA/
```

* **Train set** → Used to train the model
* **Validation set** → Used to tune the model
* **Test set** → Used to evaluate final performance


---

## ⚙️ Methodology

### 🔹 1. Data Preprocessing

* All images resized to **150 × 150 pixels**
* Pixel values normalized (**rescale = 1/255**)
* Data augmentation applied to training data:

  * Zoom transformation
  * Shear transformation
  * Horizontal flipping

👉 Purpose: Improve generalization and prevent overfitting

---

### 🔹 2. Model Architecture (CNN)

The CNN model is built using multiple layers:

1. **Conv2D (32 filters, 3×3) + ReLU**
2. **MaxPooling (2×2)**
3. **Conv2D (64 filters, 3×3) + ReLU**
4. **MaxPooling (2×2)**
5. **Conv2D (128 filters, 3×3) + ReLU**
6. **MaxPooling (2×2)**
7. **Flatten Layer**
8. **Dense Layer (128 neurons, ReLU)**
9. **Dropout Layer (0.5)** → Prevent overfitting
10. **Output Layer (Sigmoid)** → Binary classification

---

### 🔹 3. Training Configuration

* Optimizer: **Adam**
* Loss Function: **Binary Crossentropy**
* Batch Size: **16**
* Epochs: **5**

---

## 📊 Results & Analysis

### 📈 Accuracy Analysis

* Training accuracy increased from **~86% → ~93%**
* Validation accuracy fluctuated between **~68% → ~81%**

👉 Interpretation:

* Model is learning well from training data
* Performance on unseen data is unstable

---

### 📉 Loss Analysis

* Training loss consistently decreased
* Validation loss showed fluctuations and increased at final epoch

👉 Interpretation:

* Model is fitting training data effectively
* But struggling with generalization

---

## 🔍 Findings

### ✅ Strengths

* High training accuracy (**~93%**)
* CNN successfully extracted important image features
* Model is capable of distinguishing pneumonia vs normal cases

---

### ⚠️ Limitations (Overfitting)

* Validation accuracy is lower than training accuracy
* Validation loss is unstable

👉 Conclusion:
The model is **overfitting**, meaning it memorizes training data but does not generalize well to new data.

---

## 🚀 Future Improvements

To improve performance:

* Increase number of epochs with **Early Stopping**
* Tune **Dropout rate**
* Apply stronger **Data Augmentation**
* Use **Transfer Learning** (e.g., VGG16, ResNet)
* Balance dataset if class imbalance exists
* Add **Batch Normalization**

---

## 🧪 Model Evaluation

* Model tested using the **test dataset**
* Final test accuracy printed in terminal
* Performance is acceptable but can be improved

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV
* VS Code

---

## ▶️ How to Run the Project

### Step 1: Open Project

Open the project folder in **VS Code**

### Step 2: Activate Virtual Environment

```
venv\Scripts\activate
```

### Step 3: Install Dependencies

```
pip install tensorflow matplotlib numpy scikit-learn opencv-python
```

### Step 4: Run the Model

```
python main.py
```

---

## 💾 Output
* 📊 Accuracy Graph

* <img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/6d11c8a7-5f2e-4892-a053-22fbeb245303" />

* 📉 Loss Graph
* 🧪 Test Accuracy (printed in terminal)
* 💾 Saved Model file:

```
pneumonia_cnn_model.h5
```

---

## 📌 Conclusion

This project demonstrates the effectiveness of **Convolutional Neural Networks (CNNs)** in medical image classification. While the model achieves strong performance on training data, improvements are needed to enhance generalization and reduce overfitting.

This work provides a solid foundation for building more advanced AI-based healthcare diagnostic systems.

