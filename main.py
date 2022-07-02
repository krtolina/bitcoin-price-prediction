import math
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import layers

SAMPLE_SPAN = 60
price_data = yf.download('BTC-USD', start='2014-01-01', end='2022-01-01')

close_prices = price_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(SAMPLE_SPAN, len(train_data)):
    x_train.append(train_data[i - SAMPLE_SPAN:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len - SAMPLE_SPAN:, :]
x_test = []

for i in range(SAMPLE_SPAN, len(test_data)):
    x_test.append(test_data[i - SAMPLE_SPAN:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = keras.Sequential()
model.add(layers.CuDNNLSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.CuDNNLSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=3)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

data = price_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
