# app.py
import math
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout

app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r"D:\AppBulls\Stock price prediction\LSTM_API\5yearStockPrice.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.set_index('Date', inplace=True)
df = df.sort_index()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df['close']).reshape(-1, 1))

# Split into train and test
train_size = int(len(df1) * 0.75)
train_data, test_data = df1[:train_size], df1[train_size:]

# Create dataset function
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Parameters
time_step = 150
forecast_period = 180

# Function to build and train the model
def train_model():
    # Prepare training and testing data
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)
    
    # Reshape for LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Define the model
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True, input_shape=(time_step, 1))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=30, verbose=1)

    joblib.dump(model, "model.joblib")
    print("Model trained and saved as model.joblib")
    
    return model

# Load or train the model
model_file = 'model.joblib'
if os.path.exists(model_file):
    model = joblib.load("model.joblib")
    print("Loaded existing model from model.pkl")
else:
    print("No model.joblib found. Training new model...")
    model = train_model()

# Prepare test data for evaluation
x_test, y_test = create_dataset(test_data, time_step)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
test_predict = model.predict(x_test)
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate accuracy (1 - MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

test_mape = mean_absolute_percentage_error(y_test_inv[:, 0], test_predict[:, 0])
accuracy = 100 - test_mape

# API endpoint for stock price prediction
@app.route('/predict', methods=['GET'])
def predict_stock_price():
    # Forecasting for next 180 days
    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    x_input = x_input.reshape(1, time_step, 1)

    lst_output = []
    temp_input = x_input[0].copy()
    for i in range(forecast_period):
        x_input = temp_input.reshape(1, time_step, 1)
        yhat = model.predict(x_input, verbose=0)
        lst_output.extend(yhat.tolist())
        temp_input = np.roll(temp_input, -1)
        temp_input[-1] = yhat[0, 0]

    # Inverse transform the predictions
    forecast_values = scaler.inverse_transform(np.array(lst_output))

    last_date = df.index[-1]  # Last date in the CSV (e.g., 04-03-2025)
    start_date = last_date + pd.Timedelta(days=1)  # Next day (e.g., 05-03-2025)
    forecast_dates = pd.date_range(start=start_date, periods=forecast_period, freq='B')

    # Create JSON response
    forecast_dict = {str(date.date()): float(price[0]) for date, price in zip(forecast_dates, forecast_values)}
    response = {
        "forecast": forecast_dict,
        "accuracy_percentage": round(accuracy, 2)
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)