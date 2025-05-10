# IMAGE AND VIDEO PROCESSING LABORATORY (EC69211) – AUTUMN 2024

## Experiment 1: Image Scaling and Rotation Using Nearest Neighbour and Bilinear Interpolation

**Submitted by:** Unnati Singh (21EC39027)

---

## Objective

To develop a Python program that reads an image and performs image scaling and rotation using custom-implemented functions.

---

## Methodology

1. **Image Acquisition:**  
   Utilise OpenCV to read the input image file into a suitable format.

2. **Scaling:**  
   Implement custom functions for image scaling using both nearest neighbour and bilinear interpolation methods based on a given scaling factor.

3. **Rotation:**  
   Develop custom functions for image rotation using both nearest neighbour and bilinear interpolation methods based on a given rotation angle (theta).

4. **Output:**  
   Save the transformed images using OpenCV or other suitable libraries for both interpolation methods.

---

## Directory Structure

```
.
├── experiment1\_iplab.py                # Main code for scaling and rotating images
├── Input Image/
│   └── cameraman.bmp                   # Input image used for testing
├── Output Images/
│   ├── Scaling/                        # Directory to save scaled images
│   └── Rotation/                       # Directory to save rotated images
└── README.pdf                          # This README file

```

---

## Functions in the Code

### I. Scaling

1. **Nearest Neighbour Interpolation**  
   ```python
   scale_nn(scale_x, scale_y, img=cam)
   ```

* Scales the input image by the given factors using nearest neighbour interpolation.
* **Parameters:**

  * `scale_x`: Scaling factor along the x-axis.
  * `scale_y`: Scaling factor along the y-axis.
  * `img`: The input image (default is `cam`).

2. **Bilinear Interpolation**

   ```python
   scale_bl(scale_x, scale_y, img=cam)
   ```

   * Scales the input image by the given factors using bilinear interpolation.
   * **Parameters:**

     * `scale_x`: Scaling factor along the x-axis.
     * `scale_y`: Scaling factor along the y-axis.
     * `img`: The input image (default is `cam`).

---

### II. Rotation

1. **Nearest Neighbour Interpolation**

   ```python
   rotate_nn(theta, img=cam)
   ```

   * Rotates the input image by the given angle using nearest neighbour interpolation.
   * **Parameters:**

     * `theta`: Rotation angle in degrees.
     * `img`: The input image (default is `cam`).

2. **Bilinear Interpolation**

   ```python
   rotate_bl(theta, img=cam)
   ```

   * Rotates the input image by the given angle using bilinear interpolation.
   * **Parameters:**

     * `theta`: Rotation angle in degrees.
     * `img`: The input image (default is `cam`).
