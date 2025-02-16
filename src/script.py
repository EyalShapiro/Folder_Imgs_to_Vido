import cv2
import os
import sys
import numpy as np
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from constants.config import (
    get_full_path,
    FOLDER_PATH,
    OUTPUT_VIDEO,
    FPS_SINGLE,
    MAX_WIDTH,
    MAX_HEIGHT,
    ENCODE_FORMAT,
    NUM_THREADS,
)


def load_and_process_image(img_path):
    """Loads an image and returns the image along with its dimensions."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    height, width, _ = img.shape
    return img, (width, height)


def resize_keep_aspect(image, target_width, target_height):
    """Resizes the image to fit within the target size while keeping aspect ratio."""
    h, w, _ = image.shape
    scale = min(target_width / w, target_height / h)  # Determine scale factor
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_black_background(image, target_width, target_height):
    """Adds a black background to the image and centers it in the target size."""
    image = resize_keep_aspect(image, target_width, target_height)
    h, w, _ = image.shape
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Center the image on the canvas
    x_offset = (target_width - w) // 2
    y_offset = (target_height - h) // 2
    canvas[y_offset : y_offset + h, x_offset : x_offset + w] = image
    return canvas


def combine_images(img1, img2, target_width, target_height):
    """Combines two images side by side with a gap between them."""
    gap = 10  # Gap between images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape if img2 is not None else (0, 0, 3)

    # If only one image, add a black background and return it
    if img2 is None:
        return add_black_background(img1, target_width, target_height)

    # If both images can fit side by side, combine them
    if w1 + w2 + gap <= target_width:
        total_width = w1 + w2 + gap
        canvas = np.zeros((max(h1, h2), total_width, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1 + gap : w1 + gap + w2] = img2
    else:
        # Resize images if they don't fit
        scale = min(target_width / (w1 + w2 + gap), target_height / max(h1, h2))
        img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
        img2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))

        total_width = img1.shape[1] + img2.shape[1] + gap
        canvas = np.zeros(
            (max(img1.shape[0], img2.shape[0]), total_width, 3), dtype=np.uint8
        )
        canvas[: img1.shape[0], : img1.shape[1]] = img1
        canvas[
            : img2.shape[0], img1.shape[1] + gap : img1.shape[1] + gap + img2.shape[1]
        ] = img2

    return add_black_background(canvas, target_width, target_height)


def images_to_video():
    """Main function to convert images from folder into a video."""
    full_folder_path = get_full_path(FOLDER_PATH)
    image_files = sorted(
        glob(os.path.join(full_folder_path, "*.*")), key=os.path.getmtime
    )

    if not image_files:
        print("❌ No images found in folder:", full_folder_path)
        return

    # Load images in parallel to save time
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        images = list(executor.map(load_and_process_image, image_files))

    images = [(img, size) for img, size in images if img is not None]

    # Separate images into vertical and normal ones
    vertical_images = []
    normal_images = []
    for img, (w, h) in images:
        if h > w * 1.01 and max(w, h) >= 600:
            vertical_images.append(img)
        else:
            normal_images.append(img)

    # Combine both lists
    all_images = []
    i = 0
    while i < len(vertical_images):
        img1 = vertical_images[i]
        img2 = vertical_images[i + 1] if i + 1 < len(vertical_images) else None
        combined_img = combine_images(img1, img2, MAX_WIDTH, MAX_HEIGHT)
        all_images.append(combined_img)
        i += 2

    # Add normal images
    for img in normal_images:
        img = add_black_background(img, MAX_WIDTH, MAX_HEIGHT)
        all_images.append(img)

    # Shuffle all images to randomize the order
    random.shuffle(all_images)

    video_size = (MAX_WIDTH, MAX_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*ENCODE_FORMAT)
    video_writer = cv2.VideoWriter(
        get_full_path(OUTPUT_VIDEO), fourcc, FPS_SINGLE, video_size
    )

    # Write the shuffled images to the video
    for img in all_images:
        video_writer.write(img)

    video_writer.release()
    print(f"✅ Video created successfully: {get_full_path(OUTPUT_VIDEO)}")


def main():
    images_to_video()


if __name__ == "__main__":
    main()
