import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import keras
from keras import layers, models, optimizers
def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(0.2), 
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        optimizer="adam",
        loss="mae",
        metrics=["mae"]
    )

    return model

def train_model(X_train, y_train, X_val, y_val, input_dim):
    model = build_model(input_dim)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=100, 
        batch_size=64,
        callbacks=[early_stop]
    )
    return model, history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")
    return mae, rmse, r2