# Face Attendance System

A simple face recognition based attendance system with anti-spoofing support.

## Updates & Fixes

- **Improved Portability**: Fixed hardcoded Linux paths in `main.py` that were causing crashes on Windows.
- **Webcam Support**: Updated default camera index to `0` and added safety checks for frame capture.
- **Face Recognition Optimizations**:
  - Added BGR-to-RGB conversion (required by `face_recognition` library for better accuracy).
  - Implemented tolerance/sensitivity control in recognition logic.
  - Added error handling to prevent crashes when no face is detected during registration.
- **Anti-Spoofing Placeholder**: Added a `test.py` placeholder to allow the system to function even if the full anti-spoofing models are missing.
- **Code Cleanup**: Improved file handling and organized configuration variables.
- **Dependency Management**: Optimized for installation on Windows using Conda.

## Setup

### 1. Environment Setup

It is recommended to use Conda for installing dependencies, especially `dlib` which can be difficult to install via pip on Windows.

```bash
conda install -c conda-forge face_recognition dlib -y
pip install opencv-python Pillow
```

### 2. Prepare Database

Create a folder named `db` (it will be created automatically on first run) and register users through the UI.

### 3. Anti-Spoofing (Optional)

This project is designed to work with [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).
To enable full anti-spoofing, clone that repository into the project folder and download the pre-trained models.
Currently, a placeholder is used to allow the system to work without it.

## Running the App

```bash
python main.py
```

## Usage

- **Register New User**: Input a name and capture a face to add to the database.
- **Login**: Recognizes the face and logs the "in" time.
- **Logout**: Recognizes the face and logs the "out" time.
- Logs are saved in `log.txt`.
