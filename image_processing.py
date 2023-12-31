import numpy as np
from skimage import exposure
import os
from PIL import Image, ImageOps, ImageChops

# Opens image from the given path
def image_from_file_path(file_path):
    return Image.open(file_path)

# Replaces image background if empty to white
def replace_transparent_background(image):
    image_arr = np.array(image)

    has_no_alpha = len(image_arr.shape) < 3 or image_arr.shape[2] < 4
    if has_no_alpha:
        return image

    alpha1 = 0
    r2, g2, b2, alpha2 = 255, 255, 255, 255

    red, green, blue, alpha = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2], image_arr[:, :, 3]
    mask = (alpha == alpha1)
    image_arr[:, :, :4][mask] = [r2, g2, b2, alpha2]
    path = os.path.join(os.getcwd(), 'Numbers', 'Processed','transparent.png')
    if path:
        Image.fromarray(image_arr).save(path)

    return Image.fromarray(image_arr)

# shortens the image to only display the number dimmensions
def trim_borders(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    path = os.path.join(os.getcwd(), 'Numbers', 'Processed','trim.png')
    if bbox:
        if path:
            image.crop(bbox).save(path)
        return image.crop(bbox)
    if path:
        image.save(path)
    return image

# adds a white 30 pixel border to the image
def pad_image(image):
    process_image = ImageOps.expand(image, border=30, fill='#fff')
    path = os.path.join(os.getcwd(), 'Numbers', 'Processed','pad.png')
    if path:
        process_image.save(path)
    return process_image

# Turns image to grayscale
def to_grayscale(image):
    process_image = image.convert('L')
    path = os.path.join(os.getcwd(), 'Numbers', 'Processed','gray.png')
    if path:
        process_image.save(path)
    return process_image

# Inverts the image color 
# Note: image need to be black background and number area white 
def invert_colors(image):
    process_image = ImageOps.invert(image)
    path = os.path.join(os.getcwd(), 'Numbers', 'Processed','invert.png')
    if path:
        process_image.save(path)
    return process_image

# Resize the image to the dataset standard
def resize_image(image):
    processed_image = image.resize((8, 8), Image.BILINEAR)
    path = os.path.join(os.getcwd(), 'Numbers', 'Processed','resize.png')
    if path:
        processed_image.save(path)

    return processed_image

# Scale downs the image to the dataset standard
def scale_down_intensity(image):
    image_arr = np.array(image)
    image_arr = exposure.rescale_intensity(image_arr, out_range=(0, 16))
    return Image.fromarray(image_arr)

# Processes the input image to the dataset standard
def process_image(image_path):
    image = image_from_file_path(image_path)
    is_empty = not image.getbbox()
    if is_empty:
        return None

    image = replace_transparent_background(image)
    image = trim_borders(image)
    image = pad_image(image)
    image = to_grayscale(image)
    image = invert_colors(image)
    image = resize_image(image)
    image = scale_down_intensity(image)

    return np.array([
        np.array(image).flatten()
    ])
