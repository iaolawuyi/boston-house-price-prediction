import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
from plot import plot_fold_mae
from preprocessing import normalize_data
from model import build_model

# Set random seed for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

set_seed(42)

# Load and preprocess data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
train_data, test_data = normalize_data(train_data, test_data)

k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print(f"Processing fold: #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[(i+1) * num_val_samples:]
    ], axis=0)
    partial_train_targets = np.concatenate([
        train_targets[:i * num_val_samples],
        train_targets[(i+1) * num_val_samples:]
    ], axis=0)
    model = build_model()
    history = model.fit(
        partial_train_data, 
        partial_train_targets, 
        epochs=num_epochs,
        batch_size=16,
        validation_data=(val_data, val_targets),
        verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    # print(f"Fold {i} — val_mae: {val_mae:.2f}")
    # save model per fold
    model.save(f"saved_models/model_fold_{i}.keras")

num_epochs = len(all_mae_histories[0])
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
truncated_mae_hist = average_mae_history[10:]

#plotting mae fold
plot_fold_mae(truncated_mae_hist)

# Write result to file
with open("results/mae.txt", "w") as f:
    f.write(f"MAE scores per fold: {average_mae_history}\n")
    f.write(f"Mean MAE: {np.mean(average_mae_history):.2f}")