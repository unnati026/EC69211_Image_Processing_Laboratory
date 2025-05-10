# EXPERIMENT NO. 3  
## FREQUENCY DOMAIN TRANSFORMATION


---

## Overview

This experiment involves performing frequency domain transformations on grayscale images using the Fast Fourier Transform (FFT) and its inverse. The provided Python code includes functions to compute and visualise FFT and inverse FFT, as well as to perform specific image processing tasks as described in the experiment.

---

## Files Included

```

.
├── Exp3_21EC39027.py               # Python script containing the necessary functions
├── Exp3_21EC39027.ipynb            # Jupyter Notebook version of the same code
├── Experiment 3.pdf                # The problem statement
├── Exp_3_theory.pdf                # Slides with relevant theory
├── Exp3_21EC39027_report.pdf       # Experiment report with objective, methodology, results, and conclusion
├── images/                         # Directory containing input images
├── output images/                  # Directory containing output images
└── README.md                       # This README file

```

---

## Functions Overview

### 1. `FFT2D(image, magnitude_filename="magnitude_spectrum.png", phase_filename="phase_spectrum.png")`
- Computes the 2D Fast Fourier Transform of a grayscale image.
- Visualises and saves the magnitude and phase spectra as image files.

### 2. `inv_FFT2D(fft_image, output_filename="reconstructed_image.png")`
- Computes the inverse FFT of a frequency domain representation.
- Reconstructs and saves the original image from the frequency domain.

### 3. `process_image(image_path, output_filename="dip_processed.tif")`
- Applies checkerboard modulation, performs FFT, takes the complex conjugate, and applies inverse FFT followed by modulation again to reconstruct and analyse frequency behaviour.
- Normalises the result, saves it as an image, and displays it using matplotlib.
