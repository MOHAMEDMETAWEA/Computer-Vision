# Emotion Recognition using Facial Landmarks

This project implements a high-accuracy real-time emotion recognition system using facial landmarks extracted via MediaPipe and an optimized SVM Classifier.

## 🚀 Features

- **Landmark-based**: Uses 468 3D facial landmarks for detailed expression analysis.
- **Robust Normalization**: Landmarks are centered and scale-normalized (centroid-based), making the system invariant to the subject's distance and position.
- **Advanced Classifier**: Uses a Support Vector Machine (SVM) with RBF kernel and automated hyperparameter tuning via GridSearchCV for superior accuracy.
- **Temporal Smoothing**: Implements a majority-voting buffer to stabilize real-time predictions and eliminate flickering.
- **Real-time**: Highly optimized for low-latency detection on standard CPUs.

## 🛠️ Installation

1. **Clone the repository** (or navigate to the project directory).
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

- `data/`: Contains subdirectories for each emotion (`angry`, `happy`, `sad`, `surpurised`).
- `prepare_data.py`: Extracts and normalizes landmarks from raw images.
- `train_model.py`: Performs hyperparameter tuning and trains the SVM model.
- `test_model.py`: Runs real-time recognition with temporal smoothing.
- `utils.py`: Core logic for landmark extraction and geometric normalization.

## 📋 How to Use

### 1. Prepare Data

Run data preparation to regenerate `data.txt` with the new normalization logic:

```bash
python prepare_data.py
```

### 2. Train the Model

Train the optimized SVM classifier:

```bash
python train_model.py
```

### 3. Run Real-time Recognition

```bash
python test_model.py
```

## 🧩 Technical Details

- **Model**: scikit-learn `SVC` (Support Vector Classifier) with RBF kernel.
- **Normalization**: Translation (Mean Subtraction) and Scale Normalization (Max Distance Scaling).
- **Stability**: 5-frame temporal smoothing buffer using majority voting.
- **Backend**: MediaPipe Face Mesh.

## 📝 Note

After updating the code, you **must** re-run `prepare_data.py` before training, as the feature extraction logic has been upgraded for higher accuracy.
