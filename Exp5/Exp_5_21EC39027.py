# %% [markdown]
# ___

# %% [markdown]
# <center><h3>EC69211: Digital Image and Video Processing Lab</h3></center>
# <center><h4>Exp-5: Spatial Domain Filtering</h4></center>
# <center><h5>Unnati Singh | 21EC39027</h5></center>

# %%
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt

from os import listdir
from pathlib import Path

import ipywidgets as widgets
from ipywidgets import Output

import multiprocessing

# %%
def convolve2d(input_image, kernel):
    image_height, image_width = input_image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(input_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    windows = np.lib.stride_tricks.sliding_window_view(padded_image, (kernel_height, kernel_width))

    output_image = np.einsum('ijkl,kl->ij', windows, kernel)

    return output_image

# %%
def array_normalise(array):
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return (normalized_array * 255).astype(np.uint8)

# %% [markdown]
# ### Question 1: Spatial filters

# %% [markdown]
# #### Filtering:

# %% [markdown]
# Mean Filter:

# %%
def meanfilter(input_image, kernel_size=5):

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    filtered_image = convolve2d(input_image, kernel)

    return array_normalise(filtered_image)

# %% [markdown]
# Median Filter:

# %%
def medianfilter(input_image, kernel_size=5):

    height, width = input_image.shape

    filtered_image = np.zeros_like(input_image)

    for y in range(height):
        for x in range(width):
            pixel_values = []

            for ky in range(-kernel_size // 2, (kernel_size + 1) // 2):
                for kx in range(-kernel_size // 2, (kernel_size + 1) // 2):
                    ny = y + ky
                    nx = x + kx

                    if 0 <= ny < height and 0 <= nx < width:
                        pixel_values.append(input_image[ny, nx])

            median_value = np.median(np.array(pixel_values))
            filtered_image[y, x] = median_value

    return array_normalise(filtered_image)

# %% [markdown]
# Prewitt Filter:

# %%
def prewitt(input_image):

    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    gradient_x = convolve2d(input_image, prewitt_x)
    gradient_y = convolve2d(input_image, prewitt_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return array_normalise(gradient_magnitude)

# %% [markdown]
# Laplacian Filter:

# %%
def laplacian(input_image):

    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    laplacian_filtered_image = convolve2d(input_image, laplacian_kernel)

    return array_normalise(laplacian_filtered_image)

# %% [markdown]
# Sobel Filter:

# %%
def sobel(input_image):

    sobel_vertical = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_horizontal = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    sobel_diagonal1 = np.array([[2, 1, 0],
                                [1, 0, -1],
                                [0, -1, -2]])

    sobel_diagonal2 = np.array([[0, 1, 2],
                                [-1, 0, 1],
                                [-2, -1, 0]])

    sobel_kernels = {
        'horizontal': sobel_horizontal,
        'vertical': sobel_vertical,
        'diagonal1': sobel_diagonal1,
        'diagonal2': sobel_diagonal2
    }

    sobel_filtered_images = {}

    for mode, sobel_kernel in sobel_kernels.items():
        sobel_filtered_image = convolve2d(input_image, sobel_kernel)
        sobel_filtered_image = array_normalise(sobel_filtered_image)
        sobel_filtered_images[mode] = sobel_filtered_image

    return sobel_filtered_images

# %% [markdown]
# Gaussian Filter:

# %%
def gaussian_blur(input_image, kernel_size=5, sigma=1):

    def create_gaussian_kernel(kernel_size, sigma):
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        ay = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ay)

        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    gaussian_blurred_image = convolve2d(input_image, gaussian_kernel)

    return array_normalise(gaussian_blurred_image)

# %% [markdown]
# Laplacian of Gaussian:

# %%
def laplacian_of_gaussian(input_path, gaussian_kernel_size = 5, gaussian_sigma = 1.0):
    smoothed_image = gaussian_blur(input_path, kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)
    return laplacian(smoothed_image)

# %% [markdown]
# #### Stack Filtering:

# %%
def read(path):
    imgdict = {}
    for filename in os.listdir(path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            imgname = Path(filename).stem

            filepath = os.path.join(path, filename)
            imgarray = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE))

            imgdict[imgname] = imgarray

    return imgdict

# %%
def filtering(img):
    comp = {}

    comp['original'] = img

    comp['mean'] = meanfilter(img)
    comp['median'] = medianfilter(img)
    comp['prewitt'] = prewitt(img)
    comp['laplacian'] = laplacian(img)
    comp['gaussian blur'] = gaussian_blur(img)
    comp['laplacian of gaussian'] = laplacian_of_gaussian(img)

    sobelfilt = sobel(img)

    comp['sobel_h'] = sobelfilt['horizontal']
    comp['sobel_v'] = sobelfilt['vertical']
    comp['sobel_d1'] = sobelfilt['diagonal1']
    comp['sobel_d2'] = sobelfilt['diagonal2']

    return comp

# %%
def stackfiltering(path):
    images = read(path)

    pool = multiprocessing.Pool()
    outputs = pool.map(filtering, images.values())

    stack = {}

    keys = list(images.keys())

    for i in range(len(keys)):
        stack[keys[i]] = outputs[i]

    return stack

# %% [markdown]
# ### Question 2: Gaussian Unblur

# %%
def gaussian_unblur(image, sigma=1, max_iterations=100, tolerance=1e-3):

    def create_gaussian_kernel(kernel_size, sigma):
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        ay = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ay)

        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    gaussian_kernel = create_gaussian_kernel(kernel_size=5, sigma=sigma)

    I0 = gaussian_blur(image)
    Ik = np.copy(I0)

    for iteration in range(max_iterations):
        Ak = convolve2d(Ik, gaussian_kernel)
        Ak[Ak == 0] = tolerance

        Bk = I0 / Ak
        Ck = convolve2d(Bk, gaussian_kernel)

        Ik_next = array_normalise(Ik * Ck)

        diff = np.mean(np.abs(Ik_next - Ik))

        if diff < tolerance:
            print(f'Converged after {iteration + 1} iterations.')
            break

        Ik = Ik_next

    Ik[Ik > 150] = 150

    return array_normalise(Ik), I0

# %%
imgarray = np.array(cv2.imread(input("Enter image path: "), cv2.IMREAD_GRAYSCALE))

gauss = gaussian_unblur(imgarray)

plt.figure(figsize=(15, 8))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(imgarray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Blurred Image")
plt.imshow(gauss[1], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Unblurred Image")
plt.imshow(gauss[0], cmap='gray')
plt.axis('off')

plt.show()

# %% [markdown]
# ___________

# %% [markdown]
# #### UI:

# %%
# Process in my mind:
# 1. Input the path of the directory. No input will have the code take in the directory of the code file
# 2. Read the images in the directory. Return a dictionary:
#    dict =   {[image name 1]: [image array 1],
#              [image name 2]: [image array 2],
#              ...
#             }
# 3. Perform all the computations on the images and save the results as a dictionary
# 4. Dictionary format:
#    dict = {[image name 1]: {[filter 1]: [output array 1],
#                             [filter 2]: [output array 2],
#                             ...                          },
#            [image name 2]: {[filter 1]: [output array 1],
#                             [filter 2]: [output array 2],
#                             ...                          }
# 5. Have a drop down menu of all the images found in the folder
# 6. Have a drop down menu of all the filters
# 7. Display the original image and output of the selected filter on the selected image (3 different outputs in case of the sobel filter)
# 8. Have a save button to save the output.
# 9. Output image will be saved in the folder called "output" in the input directory.
# 10. Output image name: "[input image]_[filter].[extension]"
# 11. Clear the displayed image when the user selects a new option and then display the new image.

# %%
filtermap = {'Mean': 'mean',
             'Median': 'median',
             'Prewitt': 'prewitt',
             'Laplacian': 'laplacian',
             'Sobel (Horizontal)': 'sobel_h',
             'Sobel (Vertical)': 'sobel_v',
             'Sobel (Diagonal 1)': 'sobel_d1',
             'Sobel (Diagonal 2)': 'sobel_d2',
             'Gaussian Blur': 'gaussian blur',
             'Laplacian of Gaussian': 'laplacian of gaussian'}

foldername = input('Enter the path of the folder from which images are to be processed: ')
imagestack = read(foldername)

image_dropdown = widgets.Dropdown(
    options=list(imagestack.keys()),
    value=list(imagestack.keys())[0],
    description='Select image:',
)

filter_dropdown = widgets.Dropdown(
    options=list(filtermap.keys()),
    value='Mean',
    description='Select the filter:',
)

folder = widgets.Text(
    value=foldername,
    description='Folder path: ',
)

output_folder = widgets.Text(
    value='Output',
    description='Output folder:'
)

save_button = widgets.Button(
    description='Save Filtered Image',
    button_style='success'
)

output_widget = Output()

display(folder, image_dropdown, filter_dropdown, output_folder, output_widget)

out = stackfiltering(foldername)

def update(change=None):
    global first_try, foldername, out, imagestack
    f = folder.value
    image = image_dropdown.value
    filt = filter_dropdown.value

    if change and change['name'] == 'value' and (f != foldername):
        imagestack = read(f)
        image_dropdown.options = list(imagestack.keys())
        image_dropdown.value = list(imagestack.keys())[0]

        out = stackfiltering(f)

        foldername = f

    with output_widget:
        output_widget.clear_output(wait=True)

        if image in out and filtermap[filt] in out[image]:
            out_array = out[image][filtermap[filt]]
            in_array = out[image]['original']

            plt.figure(figsize=(15, 8))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(in_array, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title(f"Filtered Image (Filter: {filt})")
            plt.imshow(out_array, cmap='gray')
            plt.axis('off')
            plt.show()

            display(save_button)
        else:
            print("Error: The selected image or filter is not available.")

def save_filtered_image(b):
    image = image_dropdown.value
    filt = filter_dropdown.value
    out_array = out[image][filtermap[filt]]

    output_path = output_folder.value
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = f"{image}_{filt}.jpg"
    save_path = os.path.join(output_path, filename)

    plt.imsave(save_path, out_array, cmap='gray')
    with output_widget:
        print(f"Filtered image saved as: {save_path}")

folder.observe(update, names='value')
image_dropdown.observe(update, names='value')
filter_dropdown.observe(update, names='value')
save_button.on_click(save_filtered_image)

update()


