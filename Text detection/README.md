# Text Detection and OCR Comparison

This project provides a comprehensive comparison between three popular OCR (Optical Character Recognition) engines and includes a visualization tool for text detection.

## 🚀 Features

- **Multi-Engine Comparison**: Compare **Tesseract OCR**, **EasyOCR**, and **AWS Textract**.
- **Performance Metric**: Evaluation using **Jaccard Similarity** to measure accuracy against ground truth.
- **Data Integration**: Automatic dataset download using `kagglehub` (COCO-Text v2).
- **Visualization**: A dedicated script to visualize text detection results with bounding boxes.

## 🛠️ Installation

### 1. Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system.
  - **Windows**: Download the installer from the link above and add the installation folder to your system PATH.
  - **Linux**: `sudo apt install tesseract-ocr`

### 2. Install Dependencies

Run the following command to install the required Python libraries:

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

- `main.ipynb`: Original Google Colab notebook containing the experiments.
- `ocr_comparison_explained.py`: **(NEW)** A fully commented Python version of the notebook logic, designed for better understanding and easier modification.
- `main_plot.py`: A utility script to visualize EasyOCR results with detailed comments on data processing and drawing.
- `requirements.txt`: List of Python dependencies with explanations for each package.

## 📊 OCR Engines Overview

1. **Tesseract OCR**: An open-source, highly customizable OCR engine.
2. **EasyOCR**: A deep-learning-based OCR that works out of the box for many languages.
3. **AWS Textract**: A cloud-based service by Amazon that provides high-accuracy text extraction (requires AWS credentials).

## 🚀 Usage

### Running the Comparison

Open `main_refactored.ipynb` in your Jupyter environment. The notebook will:

- Download the COCO-Text dataset.
- Initialize the OCR engines.
- Run text detection on a subset of images.
- Calculate and display the Jaccard Similarity scores.

### Visualizing Results

To visualize text detection on an image, run:

```bash
python main_plot.py
```

*(Note: Ensure you have an image named `test.png` in the project directory, or modify the script path).*

## 🔑 AWS Textract Setup

To use AWS Textract, set your credentials as environment variables:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

If these are not set, the script will gracefully skip the Textract evaluation.

---
Created as part of CV Text Detection enhancements.
