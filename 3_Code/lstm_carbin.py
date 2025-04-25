import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = {
    "Year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    "Resident population": [7869.3, 8023, 8119.8, 8192.4, 8281.1, 8315.1, 8381.5, 8423.5, 8446.2, 8469.1, 8477.3],
    "Agriculture and forestry": [2409.2, 2736.9, 3057.8, 3228.5, 3358.6, 3636.1, 3690.6, 3568.5, 3591.6, 3726.6, 3916.8],
    "Energy supply": [904.65, 947.43, 1121.2, 1065.5, 1149.8, 1357.6, 1417.9, 1527, 1604.6, 1692.7, 1660.7],
    "Industrial consumption": [20949, 22793, 24492, 26233, 27758, 29343, 30595, 32987, 34929, 36037, 36523],
    "Transportation consumption": [1767.2, 1988.4, 2199.5, 2233.9, 2378.9, 2240.4, 2316.4, 2420.2, 2570.7, 2749.1, 2761.5],
    "Construction consumption": [15354, 17487, 19790, 22820, 25714, 28975, 32646, 35249, 38132, 41350, 43822],
    "Generate electricity": [5752.1, 6992.8, 7340, 7820.6, 6951.1, 7380.5, 7786.5, 8219.7, 8104.9, 8131.6, 7491.1],
    "heating": [204.84, 286.73, 354.09, 335.18, 382.1, 440.86, 415.01, 523.98, 1010.8, 953.12, 1004],
    "Other conversion": [-610.67, 243.37, 223.05, -968.04, -1181.7, -1275.6, -1334.3, -1635.7, -2028.8, -2376.4, -2414.7],
    "Supply loss": [454.13, 450.76, 457.5, 353.57, 500.15, 474.01, 337.92, 377.27, 370.15, 378.33, 372.9],
    "industry": [14313, 15172, 15495, 16348, 17019, 17242, 17725, 17832, 18124, 19110, 18873],
    "traffic": [1398.3, 1494.7, 1618.2, 1743.7, 1915.9, 2019.3, 2083.6, 2188, 2324.7, 2482.9, 2484.5],
    "unit": [534.58, 620.83, 690.81, 738.21, 728.32, 756.83, 794.94, 861.24, 976.14, 1039.5, 1023.1],
    "Resident life": [1148, 1205.1, 1372.1, 1469.9, 1472.5, 1566.2, 1706.3, 1862.6, 2033.7, 2070.4, 2180.6],
    "Carbon emission": [56360.05184, 65193.34223, 67502.61337, 66749.3757, 64853.27604, 66074.80995, 68526.12467, 70451.55739, 71502.00286, 74096.33108, 72633.32425]
}

df = pd.DataFrame(data)

# Extract features and target variable
features = df.drop(columns=["Year", "Carbon emission"])
target = df["Carbon emission"]

# Normalize the features
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_x.fit_transform(features)
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Convert to time series format
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 3
X_time_series, y_time_series = create_dataset(scaled_features, scaled_target, TIME_STEPS)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_time_series, y_time_series, test_size=0.2, random_state=42, shuffle=False)

# Model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, shuffle=False)

# Predictions
y_pred = model.predict(X_test)
y_train_inv = scaler_y.inverse_transform(y_train.reshape(1, -1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = scaler_y.inverse_transform(y_pred)

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plotting the actual vs predicted values
plt.plot(y_test_inv.flatten(), marker='.', label='True')
plt.plot(y_pred_inv.flatten(), 'r', marker='.', label='Predicted')
plt.ylabel('Carbon Emission')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
print('Mean Squared Error on Test Data: ', mse)

# Long-term Prediction
# Here we would ideally append the predicted values to the input features and continue predicting.
# A loop can be constructed to predict the next value, append it to features and predict again.

def predict_future(model, features, time_steps, future_steps):
    future_predictions = []
    for _ in range(future_steps):
        input_data = features[-time_steps:]
        input_data = input_data.reshape((1, time_steps, features.shape[1]))
        prediction = model.predict(input_data)
        features = np.vstack((features, prediction))
        future_predictions.append(prediction)
    return future_predictions

future_steps = 40  # predicting for 2021-2060
future_predictions_scaled = predict_future(model, scaled_features, TIME_STEPS, future_steps)
future_predictions = scaler_y.inverse_transform(future_predictions_scaled).flatten()

# Plotting future predictions
plt.figure(figsize=(10,6))
plt.plot(range(2021, 2061), future_predictions, 'r', marker='.', label='Predicted Carbon Emission')
plt.title('Future Carbon Emission Predictions')
plt.ylabel('Carbon Emission')
plt.xlabel('Year')
plt.legend()
plt.show()
