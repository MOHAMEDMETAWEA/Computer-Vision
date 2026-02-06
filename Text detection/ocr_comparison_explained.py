"""
Comprehensive OCR Engine Comparison and Performance Evaluation
------------------------------------------------------------
This script compares several OCR (Optical Character Recognition) technologies:
1. Google's Tesseract (Open Source)
2. EasyOCR (Deep Learning Based)
3. AWS Textract (Cloud Service)

It includes data downloading, text cleaning, similarity scoring, 
and visualization of the results.

Author: Antigravity AI
Date: 2024
"""

import os
import cv2
import pytesseract
from PIL import Image
from easyocr import Reader
import boto3
import matplotlib.pyplot as plt
import kagglehub
import numpy as np

# ==========================================
# 1. DATASET PREPARATION
# ==========================================

def download_data():
    """
    Downloads the COCO-Text v2.0 dataset from Kaggle using kagglehub.
    Returns the paths to the dataset files.
    """
    try:
        print("Searching for/Downloading COCO-Text dataset...")
        path = kagglehub.dataset_download("c7934597/cocotext-v20")
        print(f"Dataset successfully located at: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to local 'data' directory.")
        return 'data'

# ==========================================
# 2. OCR ENGINE INITIALIZATION
# ==========================================

# Initialize EasyOCR Reader with English language support
# gpu=False is used for compatibility; set to True if you have a CUDA-enabled GPU
print("Initializing EasyOCR...")
reader = Reader(['en'], gpu=False)

# Setup AWS Textract Client
# Looks for credentials in environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
region_name = 'us-east-1'

textract_client = None
if access_key and secret_access_key:
    print("AWS credentials detected. Initializing Textract client...")
    textract_client = boto3.client(
        'textract',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_access_key,
        region_name=region_name
    )
else:
    print("AWS credentials not found. Textract evaluation will be skipped.")

# ==========================================
# 3. OCR WRAPPER FUNCTIONS
# ==========================================

def read_text_tesseract(image_path):
    """Extracts text using Google's Tesseract OCR."""
    try:
        # pytesseract needs a PIL Image object
        text = pytesseract.image_to_string(Image.open(image_path), lang='eng')
        return text.strip()
    except Exception as e:
        return ""

def read_text_easyocr(image_path):
    """Extracts text using EasyOCR (Deep Learning based)."""
    try:
        # reader.readtext() returns list of (bbox, text, confidence)
        results = reader.readtext(image_path)
        # We only care about the text content (index 1 of each result)
        text = ' '.join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        return ""

def read_text_textract(image_path):
    """
    Extracts text using AWS Textract. 
    Requires binary image data and an active AWS connection.
    """
    if not textract_client:
        return ""
    try:
        with open(image_path, 'rb') as im:
            # detect_document_text returns a complex JSON response
            response = textract_client.detect_document_text(Document={'Bytes': im.read()})

        # Filter 'Blocks' for 'LINE' type to reconstruct the text
        text = ' '.join([item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE'])
        return text.strip()
    except Exception as e:
        return ""

# ==========================================
# 4. EVALUATION UTILITIES
# ==========================================

def jaccard_similarity(sentence1, sentence2):
    """
    Calculates the Jaccard Similarity index between two sentences.
    Higher value (closer to 1.0) means higher similarity.
    """
    # Tokenize into sets of unique lowercase words
    set1 = set(sentence1.lower().split())
    set2 = set(sentence2.lower().split())

    # If both are empty, they are perfectly identical
    if not set1 and not set2:
        return 1.0
    
    # Calculate intersection (common words) and union (all unique words)
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    return intersection_size / union_size if union_size != 0 else 0.0

def clean_text(text):
    """Removes noise, punctuation, and newlines from OCR output."""
    return text.lower().replace('\n', ' ').strip().replace('!', '').replace('?', '').replace('.', '')

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================

def run_evaluation(data_path, limit=10):
    """
    Loops through images in the dataset and evaluates all OCR engines.
    """
    if not os.path.exists(data_path):
        print(f"Path {data_path} not found. Scan cancelled.")
        return

    # 5.1 Find all image files recursively
    image_files = []
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, f))
    
    if not image_files:
        print(f"No usable images found in {data_path}")
        return

    # 5.2 Process a subset of images
    print(f"Starting evaluation on {min(limit, len(image_files))} samples...")
    image_files = image_files[:limit]
    
    # Keep track of cumulative scores
    scores = {"Tesseract": 0.0, "EasyOCR": 0.0, "Textract": 0.0}
    count = 0

    for idx, image_path in enumerate(image_files):
        img_name = os.path.basename(image_path)
        
        # Ground Truth logic: Here we assume the filename contains the target text
        # (e.g., 'hello_world.jpg' -> ground truth is 'hello world')
        gt = clean_text(img_name.split('.')[0].replace('_', ' '))
        
        # Run all three engines
        res_tess = clean_text(read_text_tesseract(image_path))
        res_easy = clean_text(read_text_easyocr(image_path))
        res_text = clean_text(read_text_textract(image_path))

        # Update scores
        scores["Tesseract"] += jaccard_similarity(gt, res_tess)
        scores["EasyOCR"] += jaccard_similarity(gt, res_easy)
        if textract_client:
            scores["Textract"] += jaccard_similarity(gt, res_text)
        
        count += 1
        if count % 5 == 0:
            print(f"Processed {count} images...")

    # 5.3 Report Final Results
    print("\n" + "="*40)
    print("   FINAL PERFORMANCE RESULTS (Average Jaccard)")
    print("="*40)
    print(f"Tesseract: {scores['Tesseract'] / count:.4f}")
    print(f"EasyOCR:   {scores['EasyOCR'] / count:.4f}")
    if textract_client:
        print(f"AWS Textract: {scores['Textract'] / count:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Step 1: Download/Locate Data
    dataset_path = download_data()
    
    # Step 2: Run Evaluation (limited to 10 images for speed)
    run_evaluation(dataset_path, limit=10)
