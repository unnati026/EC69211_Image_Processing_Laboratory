# EXPERIMENT NO. 2
---

## Overview

This experiment involves the implementation of Python functions to read, write, and manipulate BMP images. The tasks include extracting BMP header information, saving image data, and performing colour channel manipulations on an image.

---

## Files Included

```

.
├── Exp2\_21EC39027.py              # Python script with BMP operations
├── Exp2\_21EC39027.ipynb           # Jupyter Notebook version of the same code
├── Exp2\_21EC39027\_report.pdf      # Experiment report with objective, methodology, results, and conclusion
├── Input Images/                  # Directory containing input BMP images
├── Output Images/                 # Directory containing output images
└── README.pdf                     # This README file

```

---

## Functions Overview

### 1. `readBMP(path)`
- Reads a BMP file and returns the BMP header and pixel array.
- Prints key header details such as image dimensions and bit depth.

### 2. `writeBMP(outputfilename, pixelarray, size)`
- Writes the pixel array and image dimensions into a new BMP file.
- Parameters:
  - `outputfilename`: Name of the output BMP file.
  - `pixelarray`: 2D/3D pixel array of the image.
  - `size`: Tuple containing height and width.

### 3. `colourchannelmanipulation(filename, channel)`
- Modifies the specified colour channel ('R', 'G', or 'B') by setting it to zero.
- Saves the modified image to the output directory.
- Parameters:
  - `filename`: Path to the input BMP image.
  - `channel`: Colour channel to be manipulated.

---

## Notes

- This script supports **24-bit BMP images** and can also handle **8-bit grayscale** and **8-bit colour-indexed** images.
- Ensure that **input images are in BMP format**. The script will raise an error if the format is unsupported.
