# Open Set Recognition using Multi-Task Autoencoder

This project implements an Open Set Recognition (OSR) system using PyTorch. It combines an **Autoencoder** with a **Classifier** and utilizes **Triplet Loss** to shape the latent space.

## 📌 Architecture
The model uses a multi-task learning approach:
1.  **Encoder:** Maps images to a latent vector.
2.  **Decoder:** Reconstructs the image (Ensures feature quality).
3.  **Classifier:** Predicts the class (Ensures separability).

## 🚀 Key Features
* **Triplet Loss:** Clusters same-class samples together in the latent space.
* **OOD Detection:** Uses both **Reconstruction Error** and **Latent Distance** to reject unknown samples.
* **Datasets:** Trained on MNIST (In-Distribution), tested against FashionMNIST and CIFAR-10 (Out-Of-Distribution).

## 📊 Results
* **In-Distribution Accuracy (MNIST):** 96.30%
* **OOD Detection Rate:** 99.15%
* **Total Accuracy:** 97.72%

## 🛠️ Usage
1. Open the notebook `project_better_v8.ipynb`.
2. Run the training cells to calibrate thresholds.
3. Evaluate on OOD datasets.
