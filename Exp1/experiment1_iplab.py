import cv2
import numpy as np
import math

cam = cv2.imread("Input Image/cameraman.bmp", cv2.IMREAD_GRAYSCALE)


# Scaling

# Scaling using nearest neighbours interpolation:
def scale_nn(scale_x, scale_y, img=cam):
    h = img.shape[0] * scale_y
    w = img.shape[1] * scale_x

    channels = img.shape[2] if len(img.shape) == 3 else 1

    scaled = np.zeros((h, w, channels)) if len(img.shape) == 3 else np.zeros((h, w))

    for channel in range(channels):
        for i in range(h):
            for j in range(w):
                x = int(j / scale_x)
                y = int(i / scale_y)

                if x == img.shape[1]:
                    x -= 1
                if y == img.shape[0]:
                    y -= 1

                # print(i, j, "\n", x, y, "\n\n")

                if len(img.shape) == 3:
                    scaled[i, j, channel] = img[y, x, channel]
                else:
                    scaled[i, j] = img[y, x]

    return scaled


# Scaling using Bilinear interpolation:

def scale_bl(scale_x, scale_y, img=cam):
    h_orig, w_orig = img.shape[:2]
    h = int(h_orig * scale_y)
    w = int(w_orig * scale_x)
    channels = img.shape[2] if len(img.shape) == 3 else 1

    scaled = np.zeros((h, w, channels)) if channels > 1 else np.zeros((h, w))

    for channel in range(channels):
        for i in range(h):
            for j in range(w):

                x = int(j / scale_x)
                y = int(i / scale_y)

                x1 = math.floor(x)
                x2 = min(x + 1, w_orig - 1)
                y1 = math.floor(y)
                y2 = min(y + 1, h_orig - 1)

                dx = x2 - x1 if x2 != x1 else 1
                dy = y2 - y1 if y2 != y1 else 1

                w11 = (x2 - x) * (y2 - y) // dx * dy
                w12 = (x2 - x) * (y - y1) // dx * dy
                w21 = (x - x1) * (y2 - y) // dx * dy
                w22 = (x - x1) * (y - y1) // dx * dy

                if channels > 1:
                    scaled[i, j, channel] = (w11 * img[y1, x1, channel] + w12 * img[y2, x1, channel] + w21 * img[
                        y1, x2, channel] + w22 * img[y2, x2, channel])
                else:
                    scaled[i, j] = (w11 * img[y1, x1] + w12 * img[y2, x1] + w21 * img[y1, x2] + w22 * img[y2, x2])

    return scaled


# Rotation

# Rotation using nearest neighbour interpolation
def rotate_nn(theta, img=cam):
    theta_rad = np.radians(theta)

    h, w = img.shape[:2]

    channels = img.shape[2] if len(img.shape) == 3 else 1

    s = np.sin(theta_rad)
    c = np.cos(theta_rad)

    w2 = int(np.abs(w * c) + np.abs(h * s))
    h2 = int(np.abs(h * c) + np.abs(w * s))

    if channels == 1:
        rotated = np.zeros((h2, w2), dtype=img.dtype)
    else:
        rotated = np.zeros((h2, w2, channels), dtype=img.dtype)

    cx2, cy2 = w2 // 2, h2 // 2
    cx, cy = w // 2, h // 2

    for i in range(h2):
        for j in range(w2):
            y = i - cy2
            x = j - cx2

            xi = int(c * x - s * y + cx)
            yi = int(s * x + c * y + cy)

            if 0 <= xi < w and 0 <= yi < h:
                if channels > 1:
                    rotated[i, j] = img[yi, xi]
                else:
                    rotated[i, j] = img[yi, xi]

    return rotated


# Rotation using bilinear interpolation

def rotate_bl(theta, img=cam):
    theta_rad = np.radians(theta)

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    s = np.sin(theta_rad)
    c = np.cos(theta_rad)

    w2 = int(np.abs(w * c) + np.abs(h * s))
    h2 = int(np.abs(h * c) + np.abs(w * s))

    rotated = np.zeros((h2, w2)) if channels == 1 else np.zeros((h2, w2, channels))

    # Center coordinates of the new and original images
    cx2, cy2 = w2 // 2, h2 // 2
    cx, cy = w // 2, h // 2

    for i in range(h2):
        for j in range(w2):

            x = j - cx2
            y = i - cy2

            xi = c * x - s * y + cx
            yi = s * x + c * y + cy

            if 0 <= xi < w - 1 and 0 <= yi < h - 1:
                x1 = int(math.floor(xi))
                y1 = int(math.floor(yi))
                x2 = min(x1 + 1, w - 1)
                y2 = min(y1 + 1, h - 1)

                dx = x2 - x1 if x2 != x1 else 1
                dy = y2 - y1 if y2 != y1 else 1

                w11 = (x2 - xi) * (y2 - yi) / dx * dy
                w12 = (xi - x1) * (y2 - yi) / dx * dy
                w21 = (x2 - xi) * (yi - y1) / dx * dy
                w22 = (xi - x1) * (yi - y1) / dx * dy

                if channels > 1:
                    for c in range(channels):
                        rotated[i, j, c] = (
                                w11 * img[y1, x1, c] + w12 * img[y1, x2, c] + w21 * img[y2, x1, c] + w22 * img[
                            y2, x2, c])
                else:
                    rotated[i, j] = (w11 * img[y1, x1] + w12 * img[y1, x2] + w21 * img[y2, x1] + w22 * img[y2, x2])

    return rotated


image = scale_nn(2, 2)
cv2.imwrite('Output Images/Scaling/Scaled Image (Factor 2) By Nearest Neighbour Interpolation.bmp', image)

image = scale_bl(2, 2)
cv2.imwrite('Output Images/Scaling/Scaled Image (Factor 2) By Bilinear Interpolation.bmp', image)

image = rotate_nn(45)
cv2.imwrite('Output Images/Rotation/Rotated Image (45 degrees) By Nearest Neighbour interpolation.bmp', image)

image = rotate_bl(45)
cv2.imwrite('Output Images/Rotation/Rotated Image (45 degrees) By Bilinear Interpolation.bmp', image)
