import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pad_and_resize_image(image, target_size=256):
    width, height = image.size

    # Calculate the new dimensions while preserving the aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the padding
    left_padding = 0
    top_padding = 0
    right_padding = target_size - new_width
    bottom_padding = target_size - new_height

    padding = (left_padding, top_padding, right_padding, bottom_padding)

    # Pad the image using the edge pixel value
    padded_image = ImageOps.expand(resized_image, padding,
                                   fill=resized_image.getpixel((new_width - 10, new_height - 10)))
    padded_array = np.array(padded_image)

    return padded_array, new_height, new_width


def process_images(input_folder, output_folder,
                             valid_extensions=("jpg", "jpeg", "png", "bmp", "tiff", "gif")):
    os.makedirs(output_folder, exist_ok=True)
    results = []
    image_array = []
    size = []

    for file in os.listdir(input_folder):
        # file_name = os.path.splitext(file)[0]
        file_path = os.path.join(input_folder, file)

        if os.path.isfile(file_path) and file.lower().endswith(valid_extensions):
            with Image.open(file_path) as img:
                # Convert the image to RGB mode
                img_rgb = img.convert("RGB")
                padded_array, width, height = pad_and_resize_image(img_rgb)
                image_array.append(padded_array)
                size.append([width, height])

    return image_array, size


if __name__ == '__main__':
    input_folder = "images"
    input_folder_2 = "trimaps"
    output_folder = "padded_images"
    img, size = process_images(input_folder, output_folder)
    label, size_2 = process_images(input_folder_2, output_folder)
    # for padded_image in results:
    #     print(f"Original width: {padded_image.width}, Original height: {padded_image.height}")
    #     print(f"Image array shape: {padded_image.image_array.shape}")
    img = np.array(img)
    label = np.array(label)
    size = np.array(size)
    print(img[1], label[1])
    np.save("./img.npy", img)
    np.save("./label.npy", label)
    np.save("./size.npy", size)
