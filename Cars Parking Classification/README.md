# Image Classification with SVM

This project implements a simple image classification system using Support Vector Machines (SVM). It is designed to classify images into two categories: `empty` and `not_empty` (specifically tailored for applications like parking spot detection).

## Project Structure

```text
Image Classification/
├── clf-data/               # Training data organized into category folders
│   ├── empty/              # Images representing empty spots
│   └── not_empty/          # Images representing occupied spots
├── data/                   # Original video files for testing/inference
├── model/                  # Directory where the trained model is saved
│   └── model.p            # Pickled SVM model
├── main.py                 # Core script for training and evaluation
├── requirements.txt        # Project dependencies
└── LICENSE.md              # MIT License
```

## Features

- **Automated Data Loading**: Automatically crawls category subdirectories within `clf-data`.
- **Image Preprocessing**: Resizes images to a uniform 15x15 size and flattens them for the SVM classifier.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to find the optimal `gamma` and `C` values for the SVM.
- **Robust Path Handling**: Uses `argparse` for flexible input/output path management.
- **Performance Reporting**: Prints the accuracy score of the classifier on a stratified test set.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Image Classification"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model with default settings:
```bash
python main.py
```

To specify custom data or output paths:
```bash
python main.py --input_dir ./custom-data --output_path ./custom-model/model.p --test_size 0.3
```

### Arguments:
- `--input_dir`: Path to the directory containing training data (default: `./clf-data`).
- `--output_path`: Path where the trained model will be saved (default: `./model/model.p`).
- `--test_size`: Proportion of data to use for testing (default: `0.2`).
