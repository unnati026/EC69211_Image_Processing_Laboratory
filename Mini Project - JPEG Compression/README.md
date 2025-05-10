# MINI PROJECT – JPEG COMPRESSION

---

## Overview

This project implements **JPEG Compression**, a widely-used method for reducing image file sizes while preserving acceptable visual quality. The compression pipeline incorporates the following stages:
- Conversion from RGB to YCbCr colour space
- Chroma subsampling
- Block-wise Discrete Cosine Transform (DCT)
- Quantization using standard matrices
- Entropy encoding via Run-Length Encoding (RLE) and Huffman Coding

Both compression and decompression pipelines are covered, enabling visual assessment of image fidelity post-compression.

---

## Files Included

```

.
├── Mini_Project.ipynb           # Jupyter Notebook implementing full JPEG compression
├── mini_project.py              # Equivalent Python script
├── Compress_Image.ipynb         # Contains partial pipeline: till quantisation + decompression
├── Input/                       # Input images
├── Output/                      # Compressed binary and decompressed output files
├── Mini_Project_Report.pdf      # Project report with methodology, results, and analysis
├── Mini_Project_PPT.pptx        # PowerPoint presentation
├── Mini_Project_PPT.pdf         # PDF version of the presentation
└── ReadMe.pdf                   # PDF version of this README

```

---

## Functions Overview

### 1. `read_image(imagepath)`
- **Purpose:** Load an image and convert it to RGB.
- **Input:** File path.
- **Output:** RGB image (NumPy array).

---

### 2. `rgb_to_ycbcr(image)`
- **Purpose:** Convert an RGB image to YCbCr colour space.
- **Working:** Matrix multiplication and offset shifting.
- **Output:** YCbCr image (NumPy array, clamped to [0, 255]).

---

### 3. `chroma_subsampling(ycbcr_image, ratio1h=2, ratio1v=2, ratio2h=2, ratio2v=2)`
- **Purpose:** Subsample the Cb and Cr channels.
- **Working:** Horizontal and vertical decimation while keeping Y intact.
- **Output:** Y, Cb, Cr channels as separate arrays.

---

### 4. `block_formation(channel, block_size=8)`
- **Purpose:** Divide a channel into non-overlapping blocks.
- **Output:** Array of 2D blocks.

---

### 5. `dct_block(block)`
- **Purpose:** Apply Discrete Cosine Transform.
- **Output:** Frequency-domain block.

---

### 6. `quantize(block, quant_matrix)`
- **Purpose:** Reduce frequency precision using quantisation.
- **Output:** Quantised block (integers).

---

### 7. `run_length_encode(block)`
- **Purpose:** Perform zigzag scan and Run-Length Encoding.
- **Output:** List of encoded values and zero-runs.

---

### 8. `huffman(rle_data)`
- **Purpose:** Apply Huffman Coding.
- **Output:** Dictionary of codes and final bitstream (string).

---

### 9. `compress_save(image_path)`
- **Purpose:** End-to-end compression and binary file saving.
- **Working:** Invokes all previous steps and saves the compressed form using `pickle`.
- **Output:** Pickled compressed data and encoded sequence.

---

## Execution

Run either `Mini_Project.ipynb` or `Compress_Image.ipynb` for step-by-step execution and visualisation. The pipeline supports:
- Full compression and decompression
- Analysis of file size before and after
- Block visualisation at key stages (DCT, quantisation)

---

## Results & Observations

- Significant reduction in file size was achieved without major visual loss.
- Quality degradation correlates with quantisation aggressiveness.
- Huffman encoding further optimises entropy reduction.
