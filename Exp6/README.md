# EXPERIMENT NO. 6  
## HISTOGRAM EQUALIZATION AND MATCHING

---

## Overview

This experiment focuses on enhancing and analysing image contrast through **histogram equalisation** and **histogram matching**. Histogram equalisation improves the visibility of image features by redistributing pixel intensity values across the full range. Histogram matching modifies an image’s histogram to match that of a reference image, which is useful in applications requiring consistent brightness and contrast across multiple images.

---

## Files Included

```

.
├── Exp_6_21EC39027.py              # Python script with all the required functions
├── Exp_6_21EC39027.ipynb           # Jupyter Notebook for step-by-step execution
├── Input_images/                   # Directory containing original input images
├── output/                         # Directory for saving output images
├── dip_exp6.pdf                    # Lab problem statement document
└── README.md                       # This README file

```

---

## Functions Overview

### 1. `displayim(i1, i2, i3=None, t1="Original Image", t2="Equalised Image", t3=None)`
- **Purpose:** Displays up to three images side-by-side with titles.
- **Working:** Uses OpenCV to stack and label images; supports optional third image and titles.
- **Output:** Comparative visualisation of image processing results.

---

### 2. `get_hist_table(image)`
- **Purpose:** Computes histogram statistics (counts, probabilities, and CDF).
- **Working:** Processes grayscale or RGB channels separately and outputs structured data.
- **Output:** A pandas DataFrame containing pixel values, frequencies, probabilities, and cumulative distributions.

---

### 3. `plot_histogram(img)`
- **Purpose:** Plots the histogram of an image.
- **Working:** For grayscale, plots a single histogram; for RGB, generates separate plots for each channel.
- **Output:** Visual display of pixel intensity distributions.

---

### 4. `histeq(image)`
- **Purpose:** Performs histogram equalisation.
- **Working:** Enhances image contrast using cumulative distribution function (CDF); applies to the Value channel in HSV for coloured images.
- **Output:** Contrast-enhanced image with equalised histogram.

---

### 5. `histogram_matching(source_image, target_image)`
- **Purpose:** Adjusts the source image's histogram to match a reference (target) image.
- **Working:** Computes and maps CDFs to modify the intensity values.
- **Output:** Grayscale or RGB image adjusted to resemble the histogram of the target image.

---

### 6. `histogram_matching_hsv(source_image, target_image)`
- **Purpose:** Performs histogram matching in HSV colour space.
- **Working:** Matches the Value (V) component of the source to that of the target image.
- **Output:** Colour image with matched luminance histogram.