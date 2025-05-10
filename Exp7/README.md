# DIGITAL IMAGE AND VIDEO PROCESSING LAB – EXPERIMENT NO. 7  
## MORPHOLOGICAL OPERATIONS
---

## Overview

This experiment focuses on the application of fundamental morphological operations—**erosion**, **dilation**, **opening**, and **closing**—on binary images. These operations are critical in image preprocessing tasks such as noise removal, shape refinement, and object detection. The operations are performed using various structuring elements to understand their impact on the binary input.

---

## Files Included

```

.
├── Exp_07_21EC39027.py            # Python script containing functions for morphological operations
├── Exp_07_21EC39027.ipynb         # Jupyter Notebook for interactive execution
├── Input/                         # Contains binary input image (ricegrains\_mono.bmp)
├── Output/                        # Contains the resulting output images
├── Experiment_6.pdf               # Problem statement
└── README.md                      # This README file

```

---

## Functions Overview

### 1. `ErodeBinary(image, element)`
- **Purpose:** Performs erosion on the binary image.
- **Working:** Removes boundary pixels where the structuring element does not fully fit.
- **Input:** Binary image and a structuring element.
- **Output:** Eroded binary image.

---

### 2. `DilateBinary(image, element)`
- **Purpose:** Performs dilation on the binary image.
- **Working:** Expands the boundaries of objects using the structuring element.
- **Input:** Binary image and a structuring element.
- **Output:** Dilated binary image.

---

### 3. `OpenBinary(image, element)`
- **Purpose:** Applies the opening operation.
- **Working:** Erosion followed by dilation, removes small noise elements from the foreground.
- **Input:** Binary image and a structuring element.
- **Output:** Binary image after opening.

---

### 4. `CloseBinary(image, element)`
- **Purpose:** Applies the closing operation.
- **Working:** Dilation followed by erosion, fills small holes and gaps in objects.
- **Input:** Binary image and a structuring element.
- **Output:** Binary image after closing.

---

## Structuring Elements Used

1. **1 × 2 ones:** `np.ones((1, 2))`  
2. **3 × 3 ones:** `np.ones((3, 3))`  
3. **Cross-shaped:**  
   ```python
   np.array([
       [0, 1, 0],
       [1, 1, 1],
       [0, 1, 0]
   ])
    ```

4. **9 × 9 ones:** `np.ones((9, 9))`
5. **15 × 15 ones:** `np.ones((15, 15))`

---

## Results

The functions are demonstrated on the binary image `ricegrains_mono.bmp` using the above structuring elements. Each morphological operation (erosion, dilation, opening, and closing) is visualised, showing the effect of increasing structuring element size and shape. This comparison illustrates how morphological processing can enhance or suppress features depending on the structuring kernel.