import numpy as np


def readBMP(path):
    def pixels_24(pixel_array):
        pixels = []

        for i in range(0, len(pixel_array), 3):
            blue = pixel_array[i]
            green = pixel_array[i + 1]
            red = pixel_array[i + 2]

            pixels.append([red, green, blue])

        return np.array(pixels)

    def pixels_8_gray(pixel_array):
        pixels = []

        for i in range(len(pixel_array)):
            gray = pixel_array[i]

            pixels.append([gray, gray, gray])

        return np.array(pixels)

    def pixels_8_ct(pixel_array, ct):
        pixels = []

        for i in range(len(pixel_array)):
            color_index = pixel_array[i]

            blue = ct[color_index][0]
            green = ct[color_index][1]
            red = ct[color_index][2]

            pixels.append([red, green, blue])

        return np.array(pixels)

    if path.endswith('.bmp'):
        with open(path, 'rb') as image:
            img = image.read()

            # Extracting BMP header information
            width = int.from_bytes(img[18:22], 'little')
            height = int.from_bytes(img[22:26], 'little')
            size = int.from_bytes(img[2:6], 'little')
            offset = int.from_bytes(img[10:14], 'little')
            bitwidth = int.from_bytes(img[28:30], 'little')
            pixelarray = img[offset:]

            # Printing the extracted information
            print(f"Height of the image: {height} pixels")
            print(f"Width of the image: {width} pixels")
            print(f'Bit Width: {bitwidth} bits per pixel')
            print(f'File size: {size} bytes')
            print(f'Offset: {offset} bytes')

            # Handling color table (if any)
            if bitwidth == 24:
                colourtable = None
                pixels = pixels_24(pixelarray)
                print("No color table found in image data.")

            elif bitwidth == 8:
                if offset == 54:
                    colourtable = None
                    pixels = pixels_8_gray(pixelarray)
                    print("No color table found in image data.")

                else:
                    ct = img[54:offset]
                    colourtable_size = offset - 54
                    colourtable = np.array(
                        [(ct[i], ct[i + 1], ct[i + 2]) for i in range(0, colourtable_size, 4)])  # BGR Tuple
                    pixels = pixels_8_ct(pixelarray, colourtable)
                    print("Color table found in image data.")
            else:
                raise Exception("Unsupported bit depth.")

    else:
        raise Exception("File should be in .bmp format.")

    return (height, width), pixels, offset, colourtable


def writeBMP(outputfilename, pixelarray, size):
    height, width = size

    # File Header
    filetype = b'BM'
    filesize = 54 + width * height * 3
    offset = 54
    reserved = 0

    bitmapfileheader = filetype + filesize.to_bytes(4, 'little') + reserved.to_bytes(2, 'little') + reserved.to_bytes(2,
                                                                                                                      'little') + offset.to_bytes(
        4, 'little')

    # Info Header
    headersize = 40
    imagewidth = width
    imageheight = height
    planes = 1
    bitsperpixel = 24
    compression = 0
    imagesize = 0
    xpixelspermeter = 0
    ypixelspermeter = 0
    totalcolours = 0
    importantcolors = 0

    bitmapinfoheader = headersize.to_bytes(4, 'little') + imagewidth.to_bytes(4, 'little') + imageheight.to_bytes(4,
                                                                                                                  'little') + planes.to_bytes(
        2, 'little') + bitsperpixel.to_bytes(2, 'little') + compression.to_bytes(4, 'little') + imagesize.to_bytes(4,
                                                                                                                   'little') + xpixelspermeter.to_bytes(
        4, 'little') + ypixelspermeter.to_bytes(4, 'little') + totalcolours.to_bytes(4,
                                                                                     'little') + importantcolors.to_bytes(
        4, 'little')

    # Pixel Data
    pixeldata = b''
    padding_size = (4 - (width * 3) % 4) % 4
    padding = b'\x00' * padding_size

    for row in range(height):
        for col in range(width):
            pixel = pixelarray[row * width + col]
            pixeldata += int(pixel[2]).to_bytes(1, 'little') + int(pixel[1]).to_bytes(1, 'little') + int(
                pixel[0]).to_bytes(1, 'little')
        pixeldata += padding

    bmp = bitmapfileheader + bitmapinfoheader + pixeldata

    with open(outputfilename, 'wb') as f:
        f.write(bmp)


def colourchannelmanipulation(filename, channel=None):
    size, pixels, _, _ = readBMP(filename)

    if not channel:
        writeBMP(f"manipulated_{filename}.bmp", pixels, size)
        return

    newfilename = channel + '_manipulated_' + filename

    if channel.lower() == 'b':
        c = 2
    elif channel.lower() == 'g':
        c = 1
    elif channel.lower() == 'r':
        c = 0
    else:
        raise Exception('Invalid channel')

    for pixel in pixels:
        pixel[c] = 0

    writeBMP(newfilename, pixels, size)


colourchannelmanipulation('corn.bmp', 'r')
colourchannelmanipulation('corn.bmp', 'b')
colourchannelmanipulation('corn.bmp', 'g')
colourchannelmanipulation('pepper.bmp', 'r')
colourchannelmanipulation('pepper.bmp', 'b')
colourchannelmanipulation('pepper.bmp', 'g')
