import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def create_NN_model(X, y):

    # split data into train and test (25% test data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = Sequential()

    regressor.add(Dense(units=600, activation='relu'))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=100, activation='relu'))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1, activation='relu'))

    regressor.compile(optimizer=Adam(),
            loss=MeanSquaredError(),
            metrics=None)

    callback = EarlyStopping(monitor='val_loss', patience=5,
                             restore_best_weights=True)

    history = regressor.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  batch_size=50, epochs=600, verbose=True, callbacks=[callback])

    predicted = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predicted)

    y_test = y_test.reset_index(drop=True)

    symbol = '#'
    print('The model performance for testing set')
    print(f'{symbol * 40}')
    print(f'MSE is {mse}')
    print(f'{symbol * 40}')

    print("PLOT RESULTS")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(y_test, color='red')
    ax.plot(predicted, color='green')

    return regressor
