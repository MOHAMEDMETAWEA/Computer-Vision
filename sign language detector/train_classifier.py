import pickle
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


if not os.path.exists('./data.pickle'):
    print("Error: data.pickle not found. Run create_dataset.py first.")
    sys.exit(1)

try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except Exception as e:
    print(f"Error loading data.pickle: {e}")
    sys.exit(1)

# Ensure data is consistent
if not data_dict['data']:
    print("Error: Dataset is empty.")
    sys.exit(1)

# Check length consistency
lengths = [len(x) for x in data_dict['data']]
if len(set(lengths)) > 1:
    print(f"Error: Inconsistent feature lengths found: {set(lengths)}. Re-run create_dataset.py.")
    sys.exit(1)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Dataset loaded successfully!")
print(f"Total samples: {len(data)}")
print(f"Feature dimension: {data.shape[1]}")
print(f"Classes: {set(labels)}")
print(f"{'='*50}\n")

# Split data with stratification
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}\n")

# Use a more robust Random Forest with better parameters
model = RandomForestClassifier(
    n_estimators=200,  # More trees for better accuracy
    max_depth=10,      # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1          # Use all CPU cores
)

print("Training the model...")
model.fit(x_train, y_train)
print("Training complete!\n")

# Evaluate on test set
y_predict = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_predict)

print(f"{'='*50}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"{'='*50}\n")

# Cross-validation for more robust evaluation
print("Performing 5-fold cross-validation...")
cv_scores = cross_val_score(model, data, labels, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)\n")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_predict))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))
print()

# Feature importance
feature_importance = model.feature_importances_
print(f"Top 10 most important features:")
top_indices = np.argsort(feature_importance)[-10:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: {feature_importance[idx]:.4f}")

# Save the model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

print(f"\n{'='*50}")
print("Model saved to 'model.p'")
print(f"{'='*50}")
