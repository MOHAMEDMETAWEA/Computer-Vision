# Sign Language Detector 🤟

A real-time sign language detection system using MediaPipe hand tracking and machine learning. This project uses computer vision to detect hand gestures and classify them into sign language letters with high accuracy.

## Features ✨

- **Real-time Detection**: Live webcam feed with instant gesture recognition
- **High Accuracy**: Enhanced feature extraction with normalized coordinates and distance metrics
- **Confidence Scores**: Visual feedback showing prediction confidence levels
- **Prediction Smoothing**: Temporal smoothing to reduce jittery predictions
- **Visual Feedback**: Color-coded bounding boxes based on confidence (Green > 80%, Yellow > 60%, Red < 60%)
- **Robust Model**: Random Forest classifier with 200 trees and optimized parameters

## Project Structure 📁

```
sign language detector/
├── collect_imgs.py           # Collect training images from webcam
├── create_dataset.py          # Process images and extract hand landmarks
├── train_classifier.py        # Train the Random Forest model
├── inference_classifier.py    # Real-time sign language detection
├── utils.py                   # Shared utilities and configuration
├── requirements.txt           # Python dependencies
├── data/                      # Training images directory
│   ├── 0/                    # Class 0 images
│   ├── 1/                    # Class 1 images
│   └── 2/                    # Class 2 images
├── data.pickle               # Processed dataset (generated)
└── model.p                   # Trained model (generated)
```

## Installation 🔧

1. **Clone or download this repository**

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

The required packages are:

- opencv-python>=4.8.0
- mediapipe>=0.10.14
- scikit-learn>=1.3.0
- numpy>=1.24.0
- matplotlib>=3.7.0

1. **Download the Hand Landmarker Model**:
You need to download the `hand_landmarker.task` file for the new MediaPipe Tasks API:

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Or download it manually from [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it in the project root directory.

## Usage 🚀

### Step 1: Collect Training Data

Run the image collection script to gather training data for each sign:

```bash
python collect_imgs.py
```

- The script will collect 100 images for each of the 3 classes
- Press 'Q' when ready to start capturing for each class
- Images are automatically saved to `data/0/`, `data/1/`, `data/2/`

### Step 2: Create Dataset

Process the collected images and extract hand landmark features:

```bash
python create_dataset.py
```

This script:

- Detects hands using MediaPipe with high confidence threshold (0.7)
- Extracts 21 hand landmarks (42 normalized coordinates)
- Adds 5 distance features (wrist to each fingertip)
- Saves processed data to `data.pickle`

**Expected Output**:

```
Starting dataset creation...
Processing class 0...
Class 0: 95 images processed
...
Dataset creation complete!
Total images processed: 285
Dataset saved to 'data.pickle'
```

### Step 3: Train the Model

Train the Random Forest classifier:

```bash
python train_classifier.py
```

This script:

- Loads the processed dataset
- Splits data (80% train, 20% test)
- Trains a Random Forest with 200 trees
- Performs 5-fold cross-validation
- Shows detailed metrics and feature importance

**Expected Output**:

```
Dataset loaded successfully!
Total samples: 285
Training the model...
Test Accuracy: 98.25%
Mean CV accuracy: 97.50% (+/- 2.30%)
Classification Report:
...
Model saved to 'model.p'
```

### Step 4: Run Real-time Detection

Start the real-time sign language detector:

```bash
python inference_classifier.py
```

- Shows live webcam feed with hand detection
- Displays predicted sign with confidence percentage
- Color-coded bounding boxes indicate confidence level
- Press 'Q' to quit

## How It Works 🧠

### Feature Extraction

For each detected hand, the system extracts:

1. **Normalized Coordinates** (42 features):
   - 21 hand landmarks (x, y) normalized to bounding box
   - Scale-invariant representation

2. **Distance Features** (5 features):
   - Distance from wrist to thumb tip
   - Distance from wrist to index finger tip
   - Distance from wrist to middle finger tip
   - Distance from wrist to ring finger tip
   - Distance from wrist to pinky tip

**Total: 47 features per hand**

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 200
- **Max Depth**: 10 (prevents overfitting)
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2

### Prediction Smoothing

The inference system uses a voting mechanism:

- Maintains a queue of the last 5 predictions
- Returns the most common prediction
- Reduces jitter and improves stability

## Customization 🎨

### Adding More Sign Classes

1. Modify `number_of_classes` in `collect_imgs.py`:

```python
number_of_classes = 5  # Change from 3 to desired number
```

1. Update `labels_dict` in `inference_classifier.py`:

```python
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
```

1. Re-run all steps from data collection

### Adjusting Detection Confidence

In `create_dataset.py` and `inference_classifier.py`, modify:

```python
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Lower for easier detection, higher for accuracy
    min_tracking_confidence=0.5
)
```

### Changing Dataset Size

In `collect_imgs.py`:

```python
dataset_size = 200  # Increase for more training data
```

## Tips for High Accuracy 📈

1. **Good Lighting**: Ensure well-lit environment during data collection
2. **Varied Backgrounds**: Collect data with different backgrounds
3. **Hand Positions**: Vary hand positions and angles during collection
4. **Consistent Gestures**: Make gestures clearly and consistently
5. **More Data**: Collect 150-200 images per class for better accuracy
6. **Clean Data**: Remove blurry or incorrectly captured images from `data/` folders

## Troubleshooting 🔍

### MediaPipe Import Error

If you get `AttributeError: module 'mediapipe' has no attribute 'solutions'`:

```bash
pip uninstall mediapipe
pip install mediapipe==0.10.9
```

### Low Accuracy

- Collect more training data (150+ images per class)
- Ensure consistent hand gestures during collection
- Check that lighting conditions are similar during training and inference
- Verify that all classes have balanced number of samples

### Camera Not Opening

- Check if another application is using the webcam
- Try changing camera index in scripts: `cv2.VideoCapture(1)` instead of `0`
- Verify camera permissions

### Inconsistent Feature Lengths Error

- Delete `data.pickle` and re-run `create_dataset.py`
- Ensure MediaPipe version is correct (0.10.9)

## Performance Metrics 📊

With proper data collection, you can expect:

- **Training Accuracy**: 95-99%
- **Cross-validation Accuracy**: 93-98%
- **Real-time FPS**: 20-30 FPS (depending on hardware)
- **Inference Time**: ~30-50ms per frame

## Future Improvements 🚀

- [ ] Add support for dynamic gestures (motion-based signs)
- [ ] Implement deep learning models (CNN/LSTM)
- [ ] Add data augmentation for better generalization
- [ ] Support for two-handed signs
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Real-time translation to text/speech

## License 📄

This project is open source and available for educational purposes.

## Acknowledgments 🙏

- **MediaPipe**: Google's hand tracking solution
- **scikit-learn**: Machine learning library
- **OpenCV**: Computer vision library

## Contact 📧

For questions or suggestions, please open an issue in the repository.

---

**Happy Signing! 🤟**
