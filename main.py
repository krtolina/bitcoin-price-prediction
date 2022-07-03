import math
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import layers

EPOCHS = 3
BATCH_SIZE = 1
SAMPLE_SPAN = 60  # DAYS
TRAIN_TEST_RATIO = 80  # PERCENT
START_DATE = '2014-01-01'
END_DATE = '2022-01-01'

# GET DATA
price_data = yf.download('BTC-USD', start=START_DATE, end=END_DATE)
close_prices = price_data['Close'].values
training_data_len = math.ceil(len(close_prices) * TRAIN_TEST_RATIO / 100)

# SCALE DATA FROM 0 TO 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

# MAKE SAMPLES FOR TRAINING
train_data = scaled_data[0: training_data_len, :]
x_train, y_train = [], []

for i in range(SAMPLE_SPAN, len(train_data)):
    x_train.append(train_data[i - SAMPLE_SPAN:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# MAKE SAMPLES FOR TESTING
test_data = scaled_data[training_data_len - SAMPLE_SPAN:, :]
x_test, y_test = [], close_prices[training_data_len:]

for i in range(SAMPLE_SPAN, len(test_data)):
    x_test.append(test_data[i - SAMPLE_SPAN:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# MAKE AND TRAIN MODEL
model = keras.Sequential()
model.add(layers.CuDNNLSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.CuDNNLSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

# GET PREDICTIONS FROM TEST INPUT
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print("Root Mean Square Error:")
print(rmse)

# PLOT DATA
pd.options.mode.chained_assignment = None
data = price_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(16, 8))
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Actual Price', 'Price Prediction'], loc='lower right')
plt.show()
