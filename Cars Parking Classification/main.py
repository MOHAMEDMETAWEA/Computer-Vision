import os
import pickle
import argparse
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data(input_dir, categories):
    """
    Load images from the specified directory and categories.
    
    Args:
        input_dir (str): Path to the directory containing image categories.
        categories (list): List of category folder names.
        
    Returns:
        tuple: (data, labels) as numpy arrays.
    """
    data = []
    labels = []
    
    print(f"Loading data from {input_dir}...")
    
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category path {category_path} does not exist. Skipping.")
            continue
            
        for file in os.listdir(category_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, file)
                try:
                    img = imread(img_path)
                    img = resize(img, (15, 15))
                    data.append(img.flatten())
                    labels.append(category_idx)
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
    
    if not data:
        raise ValueError("No valid images found in the specified directories.")
        
    return np.asarray(data), np.asarray(labels)

def train_model(x_train, y_train):
    """
    Train an SVM model using GridSearchCV for hyperparameter tuning.
    """
    print("Training classifier with GridSearchCV...")
    classifier = SVC()
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    
    grid_search = GridSearchCV(classifier, parameters)
    grid_search.fit(x_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def main():
    parser = argparse.ArgumentParser(description="Image Classification Training Script")
    parser.add_argument('--input_dir', type=str, default='./clf-data', help='Directory containing training data')
    parser.add_argument('--output_path', type=str, default='./model/model.p', help='Path to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    categories = ['empty', 'not_empty']
    
    try:
        # Prepare data
        data, labels = load_data(args.input_dir, categories)
        
        # Train / test split
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=args.test_size, shuffle=True, stratify=labels
        )
        
        # Train classifier
        best_estimator = train_model(x_train, y_train)
        
        # Test performance
        y_prediction = best_estimator.predict(x_test)
        score = accuracy_score(y_prediction, y_test)
        
        print(f'{score * 100:.2f}% of samples were correctly classified')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        # Save model
        with open(args.output_path, 'wb') as f:
            pickle.dump(best_estimator, f)
        print(f"Model saved to {args.output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
