import os


# === Constants ===
IS_TEST = False
FOLDER_PATH = r"demo\t" if IS_TEST else r"demo\media"
OUTPUT_VIDEO = r"demo\t.mp4" if IS_TEST else r"demo\video_output.mp4"

# Video settings
FPS_SINGLE = 30  # Smooth frame rate
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
ENCODE_FORMAT = "mp4v"

# Processing settings
NUM_THREADS = 6
FADE_DURATION = 15
ZOOM_FACTOR = 1.1

# YOLO Model
YOLO_MODEL_PATH = "yolov8n.pt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # תיקיית הסקריפט


# Helper function to get full path
def get_full_path(folder_path):
    """Returns the absolute path of a folder, even if relative."""
    return (
        os.path.abspath(folder_path)
        if os.path.isabs(folder_path)
        else os.path.join(SCRIPT_DIR, folder_path)
    )
