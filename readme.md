## Task 04: Image-to-Image Translation with cGAN

### 📌 Project Overview
This project implements an image-to-image translation model using a **Conditional Generative Adversarial Network (cGAN)**, specifically the **Pix2Pix** architecture. The goal is to perform a semantic translation from a "Day" scene to a "Night" scene.
To demonstrate the concept, a synthetic dataset was generated consisting of:
 * **Source (Day):** Blue sky, green grass, and a yellow sun.
 * **Target (Night):** Dark sky, dark grass, and a grey moon.
   
### 🛠️ Technical Stack
 * **Framework:** TensorFlow / Keras
 * **Library:** Matplotlib, NumPy
 * **Model Architecture:** Simplified Pix2Pix (U-Net Generator)
 * **Optimizer:** Adam (\beta_1 = 0.5)

### 🧬 Model Architecture
The implementation utilizes a **U-Net** style generator which consists of:
 1. **Downsampling Path:** Convolutional layers that extract high-level features and reduce spatial dimensions.
 2. **Upsampling Path:** Transposed convolutional layers that reconstruct the image while utilizing skip connections from the downsampling path to preserve spatial details.
 3. **Loss Function:** Optimized using **L1 Loss** to minimize the pixel-wise difference between the generated night image and the ground-truth target.
    
### 📊 Results
The model effectively learns to map specific color distributions from the "Day" domain to the "Night" domain. Below is a comparison of the input, the AI-generated result, and the ground truth.
![WhatsApp Image 2026-04-10 at 5 31 51 PM](https://github.com/user-attachments/assets/3b08c130-e1e1-4ea8-a76b-8869453c0bce)


### 🚀 How to Run
 1. **Install dependencies:**
   ```bash
   pip install tensorflow matplotlib numpy
   
   ```
 2. **Execute the script:**
   ```bash
   math.py
   
   ```
 3. **View Results:** The script will generate a synthetic dataset, perform a training pass, and display a Matplotlib window showing the translation results.
