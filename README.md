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

## 📁 Repository Structure



RIS-BeamSelection-XAI/
│
├── beam_selection_ris.ipynb            # Single Jupyter Notebook (full pipeline)
│
├── beamforming_dataset_ris.xlsx        # Generated dataset (features + labels)
│
├── README.md                           # Project documentation
│
└── requirements.txt                    # Python dependencies


---

## 🧠 Approach

### 1. Dataset Generation

For each of `N_SAMPLES` realizations:

1. Generate:
   - Rayleigh BS → User channel `h_direct`
   - Rayleigh BS → RIS channel matrix `H_BR`
   - Rayleigh RIS → User channel `h_RU`
2. Sample a random RIS phase pattern from a small **RIS phase codebook**
3. Form the RIS-assisted component and the effective channel:
<img width="267" height="61" alt="image" src="https://github.com/user-attachments/assets/f23b335a-d4a0-4cb9-8967-64e4cf90a91c" />

4. Compute beamforming gains for all beams in the DFT codebook:

<img width="159" height="51" alt="image" src="https://github.com/user-attachments/assets/6f825eec-c119-45b6-8ae8-5199200cf468" />

   
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

<img width="283" height="71" alt="image" src="https://github.com/user-attachments/assets/771ea8ef-b09a-484f-960a-c77c46de5465" />


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


---
## 📊 Results & Visualizations

### 1. Classification Performance

- **Task:** 16-class beam index prediction from RIS-aided effective channels  
- **Final Test Loss:** `0.3197`  
- **Final Test Accuracy:** `87.07%`

This shows that a simple MLP can learn the mapping  
**effective channel → optimal DFT beam** with high reliability in a 16-beam codebook.

---

### 2. Learning Dynamics


<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/581648e3-d97f-4727-a3cc-83a09ff4f58d" />


**Key observations:**

- **Stable convergence:**  
  - Train accuracy rises from ~6% to **~91%**.  
  - Val accuracy tracks closely, saturating around **86–88%**, with no sharp divergence.
- **Smooth loss decay:**  
  - Both train and validation loss decrease monotonically from ~2.77 to **≈0.32**.  
  - The small gap between curves indicates **good generalization** and limited overfitting.
- After ~150–200 epochs, improvements are marginal → training can be **early-stopped** here in future runs.

---

### 3. Explainable AI – SHAP Feature Importance


<img width="757" height="780" alt="image" src="https://github.com/user-attachments/assets/3366ccf5-557d-47d2-8e63-3b2c7f471530" />


**What the SHAP plot shows:**

- The most influential features are mainly the **real parts** of certain antennas  
  (e.g. `h6_real`, `h8_real`, `h2_real`, `h1_real`), followed by some **imag components**
  (`h1_imag`, `h7_imag`, `h8_imag`, etc.).
- Both **real and imaginary** components contribute, confirming that the model is using
  full complex-channel information rather than over-fitting to a single dimension.
- Different antennas have different impact, which is consistent with a spatially
  selective channel where certain directions/paths (and thus beams) dominate.

**Interpretation:**

The XAI analysis confirms that:

- The network learns **physically meaningful patterns** in the RIS-modified channel.  
- Beam selection decisions are driven by a subset of spatial components with
  strongest power/phase structure, not by random noise.  
- This adds **trustworthiness** and **research value** to the model, beyond raw accuracy.

---

## 🎯 Summary

This project demonstrates:

- Physics-based RIS-aided channel simulation  
- AI-driven beam selection using deep learning  
- XAI (SHAP) to interpret beam decision mechanisms  

