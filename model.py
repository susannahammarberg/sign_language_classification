from sklearn.linear_model import LogisticRegression
import pickle

def train_model(X, y):
    """Tränar en logistisk regressionsmodell."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def save_model(model, file_path):
    """Sparar modellen till en fil."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """Laddar en sparad modell."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
