import os
import numpy as np
from preprocessing import load_and_resize, normalize_image
from features import flatten_image
from model import train_model, save_model, load_model

def load_dataset(data_dir, size=(64, 64)):
    """Laddar datasetet och förbereder data."""
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

    return np.array(X), np.array(y), label_to_index

def main():
    data_dir = "data/SSL_DATASET/Test/"
    model_path = "model.pkl"

    # Ladda dataset
    print("Laddar dataset...")
    X, y, label_to_index = load_dataset(data_dir)

    # Träna modell
    print("Tränar modell...")
    model = train_model(X, y)

    # Spara modell
    print("Sparar modell...")
    save_model(model, model_path)

    # Testa med ny bild
    print("Laddar och testar modellen...")
    test_image = "data/SSL_DATASET/Test/K/K0.jpg"  # Exempel på testbild
    img = load_and_resize(test_image)
    img = normalize_image(img)
    features = flatten_image(img).reshape(1, -1)

    # Ladda modellen och förutsäga
    loaded_model = load_model(model_path)
    prediction = loaded_model.predict(features)
    predicted_label = list(label_to_index.keys())[list(label_to_index.values()).index(prediction[0])]
    print(f"Förutsägelse: {predicted_label}")

if __name__ == "__main__":
    main()
