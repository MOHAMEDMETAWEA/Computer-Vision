# Face Anonymizer

A powerful and flexible tool for anonymizing faces in images, videos, and live webcam streams using OpenCV.

## Features

- **Multiple Input Types**: Supports individual images, entire folders, video files, and live webcam.
- **Three Anonymization Modes**:
  - `blur`: High-quality Gaussian blur.
  - `pixelate`: Classic retro pixelation effect.
  - `blackout`: Solid privacy shield (black box).
- **Customizable Intensity**: Adjust the strength of the blur or pixelation.
- **Batch Processing**: Automatically process all media files in a directory.
- **CLI Interface**: Easy to use from the command line with progress bars (`tqdm`).

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Face-anonymizer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage (Single Image)

```bash
python main.py -i data/TestFaceImage.jpg -m blur
```

### Pixelate a Video

```bash
python main.py -i path/to/video.mp4 -m pixelate -s 150
```

### Live Webcam Mode

```bash
python main.py -i webcam -m blackout
```

### Batch Process a Folder

```bash
python main.py -i path/to/images_folder -o output_folder --mode blur
```

## CLI Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Path to image/video/folder or `webcam` | N/A |
| `--output` | `-o` | Path to save the result | Auto-generated |
| `--mode` | `-m` | `blur`, `pixelate`, or `blackout` | `blur` |
| `--intensity`| `-s` | Strength of the effect (1-200) | `99` |
| `--confidence`| `-c` | Minimum detection confidence (0-1) | `0.5` |

## How it Works

The tool uses **OpenCV's Haar Cascades** for fast and reliable face detection that works across all platforms and Python versions (including 3.13+). It processes media frame-by-frame and applies the selected anonymization filter to the detected regions.

---
