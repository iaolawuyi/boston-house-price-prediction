from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
from model import build_model
from preprocessing import normalize_data

# Load and preprocess data

(train_data, _), (test_data, test_targets) = boston_housing.load_data()
_, test_data = normalize_data(train_data, test_data)

for i in range(4):
    model = keras.models.load_model(f'saved_models/model_fold_{i}.keras')
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

    print(f"test_mae_score for model {i}: {test_mae_score}")