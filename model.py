from sklearn.linear_model import LogisticRegression
import pickle

def train_model(X, y):
    """Tränar en logistisk regressionsmodell."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


# Function to create a CNN model
def create_cnn_model(input_shape, num_classes):
    """Defines and returns a CNN model for image classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def save_model(model, file_path):
    """Sparar modellen till en fil."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """Laddar en sparad modell."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
