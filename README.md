# OpenCLIP ViT-B/32 Image Encoder (NumPy Implementation)

## 📌 Overview
This repository contains the final project for the **Python Programming** course at Shenzhen University. 

The goal of this project is to reimplement the forward pass of the **OpenCLIP ViT-B/32 image encoder** using pure NumPy, without relying on deep learning frameworks like PyTorch. The implementation handles:
*   Patch splitting and linear projection (TODO-01, TODO-05a)
*   Multi-Head Attention mechanism (TODO-02, TODO-03)
*   Transformer Blocks with residual connections and MLP (TODO-04)
*   Positional embeddings and CLS token extraction (TODO-05b, TODO-06)

## 📂 File Structure
*   `image_encoder.py`: The main implementation of the model (all TODOs completed).
*   `helpers.py`: Provided helper functions (LayerNorm, GELU, attention weights, etc.).
*   `check.py`: The numerical check script to verify the output against the reference implementation.
*   `README.md`: Instructions for reproducing the results.

## ⚙️ Requirements
*   Python 3.8+
*   NumPy (The code forces `float32` precision to match the reference output)

To install the required packages, run:
```bash
pip install numpy
```

## 🚀 How to Reproduce the Results
To verify the correctness of the implementation, run the provided check script. 

**1. Check the first Transformer block:**
```bash
python check.py --stage block0
```

**2. Run the full model check (if applicable):**
```bash
python check.py
```

**Expected Output:**
The script will compare the NumPy implementation against the provided reference tensors. It should output `PASS` for all stages with numerical errors within the tolerance (e.g., `max_abs < 0.0002`).

Example output:
> `block0 PASS shape=(2, 50, 768) max_abs=7.063e-06 mean_abs=3.250e-07 (atol=0.0002, rtol=0.0002)`
> `Requested numerical checks passed.`

## 📊 Results
The program output matches the reference implementation within the given error tolerance. All test cases and numerical checks passed successfully. 

*Note: The numerical errors are mainly due to floating-point precision limits. A pure NumPy implementation is relatively inefficient without GPU parallel computing. Future work includes using GPU-accelerated frameworks like PyTorch.*

## 👨‍💻 Author
*   **Name:** Wang Qixian (王祺贤)
*   **Student ID:** 2025681077
*   **Major:** Artificial Intelligence, Shenzhen University
*   **Course:** Python Programming
