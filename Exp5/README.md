# EXPERIMENT NO. 5  
## SPATIAL DOMAIN FILTERING

---

## Overview

This experiment focuses on designing and implementing spatial filters such as Mean, Median, Prewitt, Laplacian, Sobel (in multiple orientations), Gaussian Blur, and Laplacian of Gaussian. These filters are applied to a stack of grayscale images to observe their effects on image enhancement, edge detection, and noise reduction. Additionally, the experiment introduces a custom iterative unblurring function (`gaussian_unblur`) aimed at reversing Gaussian blur effects.

---

## Files Included

```

.
├── Exp5_21EC39027.py               # Python script with all filtering functions
├── Exp5_21EC39027.ipynb            # Jupyter Notebook for interactive execution
├── input/                          # Directory with grayscale input images
├── Spatial Domain Filtering.pdf    # Lab problem statement
└── README.md                       # This README file

```

---

## Functions Overview

### 1. `meanfilter(input_image, kernel_size=5)`
- **Purpose:** Smoothens the image by averaging pixel values in a local neighbourhood.
- **Output:** Noise-reduced image with noticeable blurring.

### 2. `medianfilter(input_image, kernel_size=5)`
- **Purpose:** Reduces impulse noise while preserving image edges.
- **Output:** Cleaned image with better edge retention compared to mean filtering.

### 3. `prewitt(input_image)`
- **Purpose:** Applies Prewitt operator for edge detection in horizontal, vertical, or diagonal directions.
- **Output:** Image highlighting edges in the specified direction.

### 4. `sobel(input_image)`
- **Purpose:** Detects edges using the Sobel operator with improved noise suppression.
- **Output:** Sharper edge maps in horizontal, vertical, or diagonal orientations.

### 5. `laplacian(input_image)`
- **Purpose:** Detects edges by identifying regions of rapid intensity change.
- **Output:** Edge-detected image with no directional bias.

### 6. `gaussian_blur(input_image, kernel_size=5, sigma=1)`
- **Purpose:** Applies Gaussian smoothing to reduce image noise and fine detail.
- **Output:** Blurred image with a natural smoothing effect.

### 7. `laplacian_of_gaussian(input_path, gaussian_kernel_size=5, gaussian_sigma=1.0)`
- **Purpose:** Combines Gaussian smoothing and Laplacian edge detection.
- **Output:** Edges detected with suppressed noise artefacts.

### 8. `gaussian_unblur(image, sigma=1, max_iterations=100, tolerance=1e-3)`
- **Purpose:** Attempts to reverse Gaussian blur using iterative convolution and correction.
- **Output:** Sharpened image with reduced blur, approximating the original sharpness.