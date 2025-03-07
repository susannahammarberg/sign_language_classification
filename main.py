import os
import numpy as np
from preprocessing import load_and_resize, normalize_image
from features import flatten_image
from model import train_model, save_model, load_model
import json

import tensorflow as tf
from tensorflow.keras import layers, models

def load_dataset(data_dir, size=(64, 64)):
    # Load dataset and prepare data
    X, y = [], []
    labels = sorted(os.listdir(data_dir))  # A, B, C, ..., Z
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, file_name)
            img = load_and_resize(img_path, size)
            img = normalize_image(img)
            features = flatten_image(img)
            X.append(features)
            y.append(label_to_index[label])

    # Save label_to_index to a file
    with open('label_to_index.json', 'w') as f:
        json.dump(label_to_index, f)

    return np.array(X), np.array(y), label_to_index

def main():
    data_dir = "data/SSL_DATASET/Training/"
    model_path = "model.pkl"
 
    # # Load label_to_index from the file
    # with open('label_to_index.json', 'r') as f:
    #     label_to_index = json.load(f)

    # Load the dataset
    print("Loading dataset")
    X, y, label_to_index = load_dataset(data_dir)

    # Train the model
    print("Training the model")
    model = train_model(X, y)

    # Save model
    print("Saving the model")
    save_model(model, model_path)

    # Test with an unseen image
    print("Load and predict a test image")
    test_image = "data/SSL_DATASET/Test/K/K0.jpg"  # test image example
    img = load_and_resize(test_image)
    img = normalize_image(img)
    features = flatten_image(img).reshape(1, -1)

    # Load the model and predict the test image
    loaded_model = load_model(model_path)
    prediction = loaded_model.predict(features)
    predicted_label = list(label_to_index.keys())[list(label_to_index.values()).index(prediction[0])]
    print(f"Prediction: {predicted_label}")

if __name__ == "__main__":
    main()
