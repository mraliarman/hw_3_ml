import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Opening file
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Step 2: Encode categorical data using LabelEncoder
def encode_categorical_data(data):
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

# Step 3: Handle missing values by replacing with the mean
def handle_missing_values(data):
    data_filled = data.fillna(data.mean())
    return data_filled

# Step 4: Train-Test Split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=17, shuffle=False)

# Step 5: Implement Naive Bayes Classifier
def calculate_prior_probabilities(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return dict(zip(classes, probabilities))

def calculate_class_likelihoods(X, y, epsilon=1e-9):
    likelihoods = {}
    for class_val in np.unique(y):
        class_data = X[y == class_val]
        class_likelihood = (class_data.sum(axis=0) + 1) / (class_data.shape[0] + 2 + epsilon)  # Add epsilon for smoothing
        likelihoods[class_val] = class_likelihood
    return likelihoods

def naive_bayes_predict(X, prior_probabilities, class_likelihoods):
    predictions = []
    for sample in X.values:
        class_scores = {}
        for class_val, class_likelihood in class_likelihoods.items():
            class_score = np.log(prior_probabilities[class_val]) + np.sum(np.log(class_likelihood[sample != 0]))
            class_scores[class_val] = class_score
        predicted_class = max(class_scores, key=class_scores.get)
        predictions.append(predicted_class)
    return predictions

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Main Execution
file_path = 'Telecust1.csv'
data = read_dataset(file_path)
data_encoded = encode_categorical_data(data)
data_filled = handle_missing_values(data_encoded)

# Split attributes and labels
X = data_filled.iloc[:, 1:-1]
y = data_filled['custcat']
X_train, X_test, y_train, y_test = split_data(X, y)

# Calculate prior probabilities and class likelihoods
prior_probabilities = calculate_prior_probabilities(y_train)
class_likelihoods = calculate_class_likelihoods(X_train, y_train)

# Step 6: Implement Naive Bayes Classifier
y_pred_train = naive_bayes_predict(X_train, prior_probabilities, class_likelihoods)
y_pred_test = naive_bayes_predict(X_test, prior_probabilities, class_likelihoods)

# Step 7: Evaluate the classifier
print("Training Classification Report:")
print(classification_report(y_train, y_pred_train, zero_division=1))
plot_confusion_matrix(y_train, y_pred_train, classes=np.unique(y))

print("\nTesting Classification Report:")
print(classification_report(y_test, y_pred_test, zero_division=1))
plot_confusion_matrix(y_test, y_pred_test, classes=np.unique(y))