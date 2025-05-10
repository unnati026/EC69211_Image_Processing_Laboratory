# DIGITAL IMAGE AND VIDEO PROCESSING LAB – EXPERIMENT NO. 4  
## FREQUENCY DOMAIN FILTERING

---

## Overview

This experiment focuses on the implementation of low-pass and high-pass filters in the frequency domain using the Fast Fourier Transform (FFT). By applying these filters, we can manipulate the frequency components of grayscale images to emphasise or suppress certain features. Additionally, the experiment demonstrates the creation of a hybrid image and the denoising of an image using FFT-based techniques.

---

## Files Included

```

.
├── Exp4_21EC39027.py              # Python script with all necessary functions
├── Exp4_21EC39027.ipynb           # Jupyter Notebook version for interactive execution
├── Exp4_21EC39027_report.pdf      # Detailed experiment report
├── 4_2_EXP.pdf                    # Lab manual for reference
├── exp4_ivp.pdf                   # Detailed slides
├── input/                         # Directory containing input images
├── output/                        # Directory containing output images
└── README.md                      # This README file

```

---

## Functions Overview

### 1. `lpf(lpf_type, image_path, cutoff, order=1)`
- **Purpose:** Applies a low-pass filter to an image, allowing only low-frequency components to pass.
- **Output:** A smoothed image with high-frequency noise removed.

### 2. `hpf(hpf_type, image_path, cutoff, order=1)`
- **Purpose:** Applies a high-pass filter to an image, preserving edges and fine details.
- **Output:** An image with enhanced high-frequency features like edges and textures.

### 3. `hybrid(image1='input/einstein.png', image2='input/marilyn.png')`
- **Purpose:** Creates a hybrid image by combining high-frequency components of one image with low-frequency components of another.
- **Output:** A hybrid image that shows different features depending on the viewing distance.

### 4. `denoising(img_path)`
- **Purpose:** Denoises an image by removing frequency components that correspond to periodic or structured noise.
- **Output:** A cleaner image with reduced noise and preserved structural content.