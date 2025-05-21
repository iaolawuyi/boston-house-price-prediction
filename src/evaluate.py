from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
from model import build_model
from preprocessing import normalize_data

# Load and preprocess data

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
train_data, test_data = normalize_data(train_data, test_data)


# model = keras.models.load_model('saved_models/model_fold_0.keras')
model = build_model()
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(f"test_mae_score: {test_mae_score}")