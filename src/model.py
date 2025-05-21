from tensorflow import keras

def build_model(units=64, layers=2, learning_rate=0.001):
    model = keras.Sequential()
    for _ in range(layers):
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model

def build_model_with_batch_norm(units=64, layers=2, learning_rate=0.001):
    model = keras.Sequential()
    for _ in range(layers):
        model.add(keras.layers.Dense(units, activation=None))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        # model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model