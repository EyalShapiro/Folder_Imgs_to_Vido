## requirements.txt

```
opencv-python
numpy
```

---

## README.md

# Image to Video Converter

This script converts images from a folder into a video. It processes images in parallel, adjusts their size, combines vertical images when needed, and randomizes the order before creating the final video.

## Features

- Supports single images and vertical image pairing.
- Maintains aspect ratio with black padding.
- Multithreaded image loading for speed.
- Randomized image order in the video.

## Installation

First, install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

1. Place your images in the `media` folder.
2. Run the script:

```sh
python app.py
```

3. The output video (`video_output.mp4`) will be generated in the same directory.

## Configuration

Modify the following constants in the script if needed:

- `FOLDER_PATH`: Change the folder where images are stored.
- `OUTPUT_VIDEO`: Set a custom output video file name.
- `FPS_SINGLE`: Adjust the frame rate.
- `MAX_WIDTH` & `MAX_HEIGHT`: Change the video resolution.

## Notes

- The script automatically detects vertical images and pairs them when possible.
- If the script doesn't find images, ensure the `media` folder exists and contains images.
- The video format is `.mp4`, using `mp4v` encoding.

## License

MIT
