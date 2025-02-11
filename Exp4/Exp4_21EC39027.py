# %% [markdown]
# # <center>Image and Video Processing Lab</center>
# ## <center>Experiment 4</center>
# 
# ____________________
# 
# ### <center>Submitted by: Unnati Singh, 21EC39027</center>
# 
# _____________________

# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Output

# %% [markdown]
# ## Q1. Low-pass and High-pass filtering

# %% [markdown]
# Low pass Filter:

# %%
def lpf(lpf_type, image_path, cutoff, order=1):
  image = Image.open(image_path).convert('L')

  array = np.array(image)
  fft_2d = np.fft.fft2(array)
  fft_2d = np.fft.fftshift(fft_2d)

  mask = np.zeros(fft_2d.shape)

  for i in range(fft_2d.shape[0]):
    for j in range(fft_2d.shape[1]):
      d = np.sqrt((i - fft_2d.shape[0]//2)**2 + (j - fft_2d.shape[1]//2)**2)
      if lpf_type.lower() == 'butterworth':
        den = 1 + (d/cutoff)**(2*order)
        mask[i][j] = 1/den

      elif lpf_type.lower() == 'ideal':
        if d <= cutoff:
          mask[i][j] = 1

      elif lpf_type.lower() == 'gaussian':
        mask[i][j] = np.exp(-(d**2)/(2*cutoff**2))
        
      else:
        raise TypeError("Invalid LPF provided.")

  fft_shifted = mask*fft_2d

  fft_unshift = np.fft.ifftshift(fft_shifted)

  final = np.real(np.fft.ifft2(fft_unshift))

  maximum = np.max(final)
  minimum = np.min(final)

  if maximum == minimum:
    diff = 1

  else:
    diff = maximum - minimum

  final = (255*(final - minimum)/diff).astype(np.uint8)

  return fft_2d, mask, fft_shifted, final

# %%
lpf_dropdown = widgets.Dropdown(
    options=['Butterworth', 'Gaussian', 'Ideal'],
    value='Butterworth',
    description='LPF Type:',
)

cutoff_slider = widgets.IntSlider(
    value=50,
    min=10,
    max=1000,
    step=10,
    description='Cutoff:',
    continuous_update=False
)

image_text = widgets.Text(
    value=input('Enter name of image file'),
    description='Image File:',
)

order_label = widgets.Label(
    value='Order (only for Butterworth filter):'
)
order_input = widgets.IntText(
    value=1,
    disabled=False,
    # layout=widgets.Layout(width='100px')
)

output_widget = Output()

display(lpf_dropdown, cutoff_slider, image_text, order_label, order_input, output_widget)

def update_lpf(change):
    l = lpf_dropdown.value
    image = image_text.value
    cutoff = cutoff_slider.value
    order = order_input.value

    order_input.disabled = (l != 'Butterworth')
    
    with output_widget:
        output_widget.clear_output(wait=True)
        
        original, filter_lpf, fft, final_image = lpf(l, image, cutoff, order)

        orig = 20 * np.log(np.abs(original) + 1e-6)
        filter_spec = 20 * np.log(np.abs(filter_lpf) + 1e-6)
        new = 20 * np.log(np.abs(fft) + 1e-6)

        plt.figure(figsize=(20, 8))
        
        plt.subplot(1, 4, 1)
        plt.title("FFT Magnitude Spectrum of Original Image")
        plt.imshow(orig, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("FFT Magnitude Spectrum of the Filter")
        plt.imshow(filter_spec, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("FFT Magnitude Spectrum of Filtered Image")
        plt.imshow(new, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("Filtered Image")
        plt.imshow(final_image, cmap='gray')
        plt.axis('off')
        
        plt.show()

lpf_dropdown.observe(update_lpf, names='value')
cutoff_slider.observe(update_lpf, names='value')
order_input.observe(update_lpf, names='value')
image_text.observe(update_lpf, names='value')

update_lpf(None)

# %% [markdown]
# High Pass Filter:

# %%
def hpf(hpf_type, image_path, cutoff, order=1):
  image = Image.open(image_path).convert("L")

  array = np.array(image)
  fft_2d = np.fft.fft2(array)
  fft_2d = np.fft.fftshift(fft_2d)

  mask = np.zeros(fft_2d.shape)

  for i in range(fft_2d.shape[0]):
    for j in range(fft_2d.shape[1]):
      d = np.sqrt((i - fft_2d.shape[0]//2)**2 + (j - fft_2d.shape[1]//2)**2)
      if hpf_type.lower() == 'butterworth':
        if d == 0:
          den = 1
      
        else:
          den = 1 + (cutoff/d)**(2*order)
        mask[i][j] = 1/den

      elif hpf_type.lower() == 'ideal':
        if d > cutoff:
          mask[i][j] = 1

      elif hpf_type.lower() == 'gaussian':
        mask[i][j] = 1 - np.exp(-(d**2)/(2*cutoff**2))
        
      else:
        raise TypeError("Invalid LPF provided.")

  fft_shifted = mask*fft_2d

  fft_unshift = np.fft.ifftshift(fft_shifted)

  final = np.real(np.fft.ifft2(fft_unshift))

  maximum = np.max(final)
  minimum = np.min(final)

  if maximum == minimum:
    diff = 1

  else:
    diff = maximum - minimum

  final = (255*(final - minimum)/diff).astype(np.uint8)

  return fft_2d, mask, fft_shifted, final

# %%
hpf_dropdown = widgets.Dropdown(
    options=['Butterworth', 'Gaussian', 'Ideal'],
    value='Butterworth',
    description='LPF Type:',
)

cutoff_slider = widgets.IntSlider(
    value=20,
    min=10,
    max=1000,
    step=10,
    description='Cutoff:',
    continuous_update=False
)

image_text = widgets.Text(
    value=input('Enter name of image file'),
    description='Image File:',
)

order_label = widgets.Label(
    value='Order (only for Butterworth filter):'
)
order_input = widgets.IntText(
    value=1,
    disabled=False,
    layout=widgets.Layout(width='100px')
)

output_widget = Output()

display(hpf_dropdown, cutoff_slider, image_text, order_label, order_input, output_widget)

def update_hpf(change):
    h = hpf_dropdown.value
    image = image_text.value
    cutoff = cutoff_slider.value
    order = order_input.value

    order_input.disabled = (h != 'Butterworth')
    
    with output_widget:
        output_widget.clear_output(wait=True)
        
        original, filter_hpf, fft, final_image = hpf(h, image, cutoff, order)

        orig = 20 * np.log(np.abs(original) + 1e-6)
        filter_spec = 20 * np.log(np.abs(filter_hpf) + 1e-6)
        new = 20 * np.log(np.abs(fft) + 1e-6)

        plt.figure(figsize=(20, 8))
        
        plt.subplot(1, 4, 1)
        plt.title("FFT Magnitude Spectrum of Original Image")
        plt.imshow(orig, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("FFT Magnitude Spectrum of the Filter")
        plt.imshow(filter_spec, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("FFT Magnitude Spectrum of Filtered Image")
        plt.imshow(new, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("Filtered Image")
        plt.imshow(final_image, cmap='gray')
        plt.axis('off')
        
        plt.show()

hpf_dropdown.observe(update_hpf, names='value')
cutoff_slider.observe(update_hpf, names='value')
order_input.observe(update_hpf, names='value')
image_text.observe(update_hpf, names='value')

update_hpf(None)

# %% [markdown]
# ## Q2. Creating the Hybrid Image

# %%
def hybrid(image1='input/einstein.png', image2='input/marilyn.png'):
    i1 = np.array(Image.open(image1).convert('L'))
    i2 = np.array(Image.open(image2).convert('L'))
    shape = i1.shape

    _, _, image2_low, _ = lpf('gaussian', image2, cutoff=20, order=2)

    _, _, image1_high, _ = hpf('gaussian', image1, cutoff=20, order=2)

    hybrid = np.fft.ifftshift(image2_low + image1_high)

    hybrid_image = np.real(np.fft.ifft2(hybrid, shape))

    minimum = np.min(hybrid_image)
    maximum = np.max(hybrid_image)

    hybrid_image = 255 * (hybrid_image - minimum)/(maximum - minimum)

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 3, 1)
    plt.title("Image 1")
    plt.imshow(i1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Image 2")
    plt.imshow(i2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Hybrid")
    plt.imshow(hybrid_image, cmap='gray')
    plt.axis('off')
    plt.show()

    imgsave = Image.fromarray(hybrid_image.astype(np.uint8))
    imgsave.save('output/hybrid.jpg')

    return hybrid_image

# %%
array = hybrid()

# %% [markdown]
# ## Q3. Denoising

# %%
def denoising(img_path):
    image = Image.open(img_path).convert('L')
    image = np.array(image, dtype=float)

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-6)

    radius = 10
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2 
    mask = np.ones((rows, cols))
    mask2 = np.copy(dft_shift)
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
    mask[dist_from_center <= radius] = 0
    mask2[dist_from_center > radius] = 0
    magnitude_spectrum_filtered = magnitude_spectrum * mask

    size = 2
    threshold = 250
    brightest_spots = []

    for r in range(size, rows - size):
        for c in range(size, cols - size):
            patch = magnitude_spectrum_filtered[r-size:r+size+1, c-size:c+size+1]
            if magnitude_spectrum_filtered[r, c] == np.max(patch):
                if magnitude_spectrum_filtered[r, c] > threshold:
                    brightest_spots.append((r, c))

    line_width = image.shape[0]

    median_value = np.median(magnitude_spectrum)
    magnitude_spectrum_replaced = np.copy(magnitude_spectrum_filtered)

    for spot in brightest_spots:
        rr, cc = spot
        
        start_row = max(rr - line_width // 2, 0)
        end_row = min(rr + line_width // 2 + 1, rows)
        magnitude_spectrum_replaced[start_row:end_row, cc] = median_value
        
        start_col = max(cc - line_width // 2, 0)
        end_col = min(cc + line_width // 2 + 1, cols)
        magnitude_spectrum_replaced[rr, start_col:end_col] = median_value

    magnitude_replaced = np.exp(magnitude_spectrum_replaced / 20)
    phase_replaced = np.angle(dft_shift)
    dft_replaced = magnitude_replaced * np.exp(1j * phase_replaced)

    dft_replaced += mask2

    magnitude_spectrum_replaced = 20 * np.log(np.abs(dft_replaced) + 1e-6)


    image_replaced = np.fft.ifft2(np.fft.ifftshift(dft_replaced))
    image_replaced = np.abs(image_replaced) 

    maximum = np.max(image_replaced)
    minimum = np.min(image_replaced)

    image_replaced = (255 * (image_replaced - minimum)/(maximum - minimum))

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 3, 1)
    plt.title('Original Magnitude Spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Replaced Lines Magnitude Spectrum')
    plt.imshow(magnitude_spectrum_replaced, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Processed Image from Replaced Spectrum')
    plt.imshow(image_replaced, cmap='gray')
    plt.axis('off')

    plt.show()

    magnitude_spectrum_img = Image.fromarray((magnitude_spectrum).astype(np.uint8))
    magnitude_spectrum_replaced_img = Image.fromarray((magnitude_spectrum_replaced).astype(np.uint8))
    image_replaced_img = Image.fromarray((image_replaced).astype(np.uint8))

    if not os.path.exists('output'):
        os.mkdir('output')

    magnitude_spectrum_img.save('output/original_magnitude_spectrum'+img_path)
    magnitude_spectrum_replaced_img.save('output/replaced_lines_magnitude_spectrum'+img_path)
    image_replaced_img.save('output/processed_image'+img_path)

    return image_replaced

# %%
den = denoising(input("Enter the name of the image to be denoised: "))

# %%



