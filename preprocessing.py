import cv2
import os

def load_and_resize(image_path, size=(64, 64)):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    # resize image
    img = cv2.resize(img, size)
    return img

def normalize_image(img):
    # Normalise pixel values to between 0 and 1
    return img / 255.0
