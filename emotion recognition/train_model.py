import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data from the text file
data_file = "data.txt"
if not os.path.exists(data_file):
    print(f"Error: Data file '{data_file}' not found. Please run prepare_data.py first.")
    exit()

data = np.loadtxt(data_file)
print(f"Loaded {data.shape[0]} samples from {data_file}.")

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Labels are the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)

# Initialize the Classifier - Using SVM with RBF kernel as it's 
# often more effective for geometric features like landmarks.
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf']
}

print("Searching for the best hyperparameters...")
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model_path = './model'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Model saved to {model_path}")

