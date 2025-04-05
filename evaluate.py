import os
import numpy as np
import random
from tensorflow.keras.models import load_model
from preprocess import audio_to_spectrogram
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define paths
MODEL_FILE = "deepfake_detector.h5"  # Update this with your model file path
DEVELOPMENT_DATASET_FOLDER = "<path-to-dataset>"  # Update this path
LABELS_FILE = "ASVspoof5.dev.track_1.tsv"  # Update this with actual labels file(this is the one in the ASVspoof 5 website for the developement dataset)

# Load model
model = load_model(MODEL_FILE)

def load_labels(labels_file):
    """ Load ground truth labels from the TSV file """
    labels = {}
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()  # Ensure proper tab separation
            if len(parts) < 10:  # Avoid incorrect formatting
                continue  
            file_name = parts[1] + ".flac"  # Assuming the 2nd column is the file identifier
            label = 1 if parts[8].strip().lower() == "bonafide" else 0  # 1 for bonafide, 0 for spoof
            labels[file_name] = label
    return labels


import re

def numeric_sort(file_list):
    return sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

def evaluate_model(sample_size=100):
    """ Evaluate the model on the first N files in the dataset """
    labels = load_labels(LABELS_FILE)

    dataset_files = numeric_sort([f for f in os.listdir(DEVELOPMENT_DATASET_FOLDER) if f.endswith(".flac")])
    if len(dataset_files) == 0:
        print("No valid files found in the dataset folder!")
        return

    # Select first `sample_size` files
    sample_files = dataset_files[:sample_size]

    print(f"Selected {len(sample_files)} files")
    print("First 10 selected files:", sample_files[:10])  # Debugging output

    predictions, actual_labels = [], []

    for file_name in sample_files:
        file_path = os.path.join(DEVELOPMENT_DATASET_FOLDER, file_name)

        try:
            spectrogram = audio_to_spectrogram(file_path)[np.newaxis, ..., np.newaxis]
            prediction = model.predict(spectrogram)[0]
            print(prediction)
            predicted_label = 1 if prediction >= 0.999 else 0

            predictions.append(predicted_label)
            actual_labels.append(labels[file_name])
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not predictions:
        print("No valid predictions were made. Check input data and model output.")
        return
    
    compute_metrics(predictions, actual_labels)

def compute_metrics(predictions, actual_labels):
    """ Compute and print performance metrics """
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, zero_division=0)
    recall = recall_score(actual_labels, predictions, zero_division=0)
    f1 = f1_score(actual_labels, predictions, zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate_model(sample_size=300)  # Adjust sample size if needed
