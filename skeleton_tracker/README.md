# Skeleton Tracker

A command-line tool for skeleton tracking using YOLOv8 Pose.

## Features
- Tracks "Hips," "Head," "Hands," and "Feet" zones.
- Real-time visualization with semi-transparent overlays.
- Robust video processing with automatic codec fallback.
- Fully managed with `uv`.

## Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
uv sync
```

## Usage

You can run the tracker using `uv run track` with various options:

```bash
uv run track --input path/to/video.mp4 --output output.mp4
```

### Options:
- `--input`: Path to input video file (default: `../assets/piscine.mp4`).
- `--output`: Path to save output video file (default: `output.mp4`).
- `--model`: YOLOv8 pose model to use (default: `yolov8n-pose.pt`).
- `--conf`: Confidence threshold for keypoints (default: `0.5`).
- `--no-show`: Run without displaying the tracking window (useful for headless environments).

### Example:
```bash
uv run track --input assets/my_video.mp4 --conf 0.7 --no-show
```

## Development

The project uses a `src-layout`. The main logic is located in `src/skeleton_tracker/track.py`.
