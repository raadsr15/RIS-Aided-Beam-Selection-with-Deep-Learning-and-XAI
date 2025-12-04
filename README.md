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
