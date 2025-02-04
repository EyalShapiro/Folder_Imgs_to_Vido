import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # תיקיית הסקריפט


def get_full_path(folder_path):
    """Returns the absolute path of a folder, even if relative."""
    return (
        os.path.abspath(folder_path)
        if os.path.isabs(folder_path)
        else os.path.join(SCRIPT_DIR, folder_path)
    )
