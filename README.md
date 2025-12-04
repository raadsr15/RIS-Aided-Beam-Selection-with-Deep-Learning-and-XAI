# RIS-Aided Beam Selection with Deep Learning and XAI

This repository implements a **6G-style RIS-aided beam selection system** with:

- Physics-based channel simulation (direct + RIS-assisted paths)
- Deep learning–based beam prediction (PyTorch MLP)
- Explainable AI (SHAP) to interpret which channel components drive decisions

The project is designed as a **research-flavoured demo** aligned with current work on
RIS-aided massive MIMO, ML for wireless communications, and 6G beam management.


---

## 🌐 Problem Overview

We consider a downlink system where:

- A Base Station (BS) has **Nt = 8 antennas**
- A Reconfigurable Intelligent Surface (RIS) has **M = 16 passive elements**
- The BS uses a **DFT beamforming codebook** with **K = 16 beams**
- The user receives both:
  - a **direct BS → User** path
  - a **RIS-assisted BS → RIS → User** path

For each channel realization, we form the **effective channel**

---

## 🧠 Approach

### 1. Dataset Generation

For each of `N_SAMPLES` realizations:

1. Generate:
   - Rayleigh BS → User channel `h_direct`
   - Rayleigh BS → RIS channel matrix `H_BR`
   - Rayleigh RIS → User channel `h_RU`
2. Sample a random RIS phase pattern from a small **RIS phase codebook**
3. Form the **RIS-assisted component** and the effective channel:
   \[
   h_{\text{eff}} = h_{\text{direct}} + h_{\text{RU}} \Theta H_{\text{BR}}
   \]
4. Compute beamforming gains for all beams in the DFT codebook:
   \[
   g_k = |h_{\text{eff}} w_k|^2
   \]
5. Label = `argmax_k g_k` (best beam index)
6. Features = concatenation of real and imaginary parts of \(h_{\text{eff}}\):
   \[
   x = [\Re(h_1),\dots,\Re(h_{Nt}),\Im(h_1),\dots,\Im(h_{Nt})]
   \]

The resulting dataset is:

- `X`: shape `(N_SAMPLES, 2 * Nt)`
- `y`: shape `(N_SAMPLES,)`, integer beam indices in `[0, K-1]`

The dataset is also exported as:

```text
data/beamforming_dataset_ris.xlsx
```



---

## 🔧 2. Deep Learning Model (Beam Predictor)

A lightweight PyTorch **MLP classifier** is trained to map the effective channel  
(real + imaginary components) to the optimal BS beam index.

### **Model Architecture**
- Input dimension: `2 * Nt = 16`
- Two hidden layers (128 → 128 ReLU)
- Output: `K = 16` beam logits
- Loss: CrossEntropy
- Optimizer: Adam (lr = 1e-3)

### **What the Model Learns**
The neural network approximates the function:

\[
f: \mathbb{R}^{16} \rightarrow \{0, 1, \dots, 15\}
\]

assigning each channel realization to the **best beam** from the DFT codebook.

---

## 📉 3. Training & Evaluation

The dataset is divided into:

- **70% Train**
- **15% Validation**
- **15% Test**

During training, we collect and plot:

- Training vs Validation Accuracy  
- Training vs Validation Loss  

The best-performing model (highest validation accuracy) is saved.

### **Final Output**
- `final_test_accuracy`
- `final_test_loss`
- Saved model weights: `models/best_model.pt`

---

## 🔍 4. Explainable AI (SHAP)

We integrate **SHAP KernelExplainer** to interpret the trained MLP.

### **Process**
1. Summarize background dataset using K-Means  
2. Compute SHAP values for 300 test samples  
3. Average absolute SHAP values over all beams  
4. Plot global feature importance

### **Why XAI?**
- Shows which BS antenna components matter most  
- Highlights how RIS modifies channel importance  
- Adds transparency and research depth to the ML model

Output includes:

- **Global SHAP Importance Plot**
- (Optional) Per-sample explanations

---


## 🎯 Summary

This project demonstrates:

- Physics-based RIS-aided channel simulation  
- AI-driven beam selection using deep learning  
- XAI (SHAP) to interpret beam decision mechanisms  

