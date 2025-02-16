import cv2
import os
import numpy as np
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from ultralytics import YOLO
from PIL import Image
from model.HeicConverterPng import HeicConverterPng

# Import constants
from constants.config import (
    FOLDER_PATH,
    OUTPUT_VIDEO,
    FPS_SINGLE,
    MAX_WIDTH,
    MAX_HEIGHT,
    ENCODE_FORMAT,
    NUM_THREADS,
    FADE_DURATION,
    ZOOM_FACTOR,
    YOLO_MODEL_PATH,
    get_full_path,
)

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)


def load_and_process_image(img_path):
    """Loads an image and returns the image along with its dimensions."""
    if img_path.lower().endswith(".gif"):
        try:
            gif = Image.open(img_path)
            gif = gif.convert("RGB")  # ממיר לפרופיל צבע RGB
            img_path = img_path.replace(".gif", ".png")  # שומר כ-PNG
            gif.save(img_path)
            print(f"🔄 Converted GIF to PNG: {img_path}")
        except Exception as e:
            print(f"⚠️ Failed to convert GIF: {img_path}, Error: {e}")
            return None, None
    elif img_path.lower().endswith(".heic"):
        convertHeic = HeicConverterPng()
        convertHeic.convert_heic_to_png_and_replace(img_path)

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Failed to load image: {img_path}")
        return None, None
    height, width, _ = img.shape
    return img, (width, height)


def resize_image(image):
    """Resizes the image to fit the video frame size."""
    return cv2.resize(image, (MAX_WIDTH, MAX_HEIGHT))


def add_zoom_effect(image):
    """Adds a smooth zoom-in effect to the image."""
    h, w, _ = image.shape
    zoom_w = int(w * ZOOM_FACTOR)
    zoom_h = int(h * ZOOM_FACTOR)

    resized = cv2.resize(image, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
    x_start = (zoom_w - w) // 2
    y_start = (zoom_h - h) // 2
    return resized[y_start : y_start + h, x_start : x_start + w]


def analyze_image_colors(image, num_clusters=3):
    """Analyzes the dominant colors in an image using K-Means clustering."""
    img_resized = cv2.resize(image, (100, 100))  # Reduce size for faster computation
    img_data = img_resized.reshape((-1, 3))  # Convert image to pixel array

    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(img_data)

    dominant_color = np.mean(kmeans.cluster_centers_, axis=0)
    return tuple(dominant_color)


def detect_objects_yolo(image):
    """Uses YOLOv8 to detect objects in an image and returns the main object category."""
    results = yolo_model(image)

    if not results[0].boxes:
        return "unknown"

    return results[0].names[int(results[0].boxes.cls[0])]


def crossfade_images(img1, img2, frames=FADE_DURATION):
    """Creates a crossfade transition between two images."""
    height, width, _ = img1.shape
    img2 = cv2.resize(img2, (width, height))

    fade_frames = []
    for alpha in np.linspace(0, 1, frames):
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        fade_frames.append(blended)

    return fade_frames


def intelligent_sort_images(image_files):
    """Sorts images based on color similarity and object detection."""
    image_data = []

    for img_path in image_files:
        img, _ = load_and_process_image(img_path)
        if img is None:
            continue

        color = analyze_image_colors(img)
        objects = detect_objects_yolo(img)
        image_data.append((img_path, color, objects))

    image_data.sort(key=lambda x: (x[2], x[1]))

    return [img[0] for img in image_data]


def images_to_video():
    """Main function to convert images into a video with smooth transitions."""
    full_folder_path = get_full_path(FOLDER_PATH)
    image_files = glob(os.path.join(full_folder_path, "*.*"))
    # image_files = [f for f in image_files if not f.lower().endswith(".gif")]

    if not image_files:
        print("❌ No images found in folder:", full_folder_path)
        return

    sorted_images = intelligent_sort_images(image_files)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        images = list(executor.map(load_and_process_image, sorted_images))

    images = [(img, size) for img, size in images if img is not None]

    if not images:
        print("❌ No valid images to process.")
        return

    final_frames = []

    for i in range(len(images) - 1):
        img1, _ = images[i]
        img2, _ = images[i + 1]

        img1 = resize_image(add_zoom_effect(img1))
        img2 = resize_image(add_zoom_effect(img2))

        final_frames.append(img1)

        transition_frames = crossfade_images(img1, img2)
        final_frames.extend(transition_frames)

    final_frames.append(resize_image(images[-1][0]))

    if not final_frames:
        print("❌ No frames to write into the video!")
        return

    print(f"🎞️ Total frames in video: {len(final_frames)}")

    video_size = (MAX_WIDTH, MAX_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*ENCODE_FORMAT)
    video_writer = cv2.VideoWriter(
        get_full_path(OUTPUT_VIDEO), fourcc, FPS_SINGLE, video_size
    )

    for frame in final_frames:
        video_writer.write(frame)

    video_writer.release()
    print(
        f"✅ AI-powered smooth video created successfully: {get_full_path(OUTPUT_VIDEO)}"
    )


def main():
    images_to_video()


if __name__ == "__main__":
    main()
