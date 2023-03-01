import pandas as pd
import autokeras as ak
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skpp
import numpy as np

# Initialize scaler
scaler = skpp.MinMaxScaler()

# Read in the csv file
data = pd.read_csv("G:\\Other computers\\KUNA DESKTOP\\Creative Cloud Files\\python projs\\galottery\\fullcomb\\combined.csv", header=0)

# Remove missing data
data = data.dropna()

# Convert the "DRAW" column to numerical format
draw_mapping = {"MIDDAY": 0, "EVENING": 1, "NIGHT": 2}
data["DRAW"] = data["DRAW"].map(draw_mapping).astype("float64")

# Convert "DATE" column to datetime
data['DATE'] = pd.to_datetime(data['DATE'])
data['DATE'] = data['DATE'].apply(lambda x: x.value).astype("float64")

# Format the "WINNING NUMBERS" column
data['WINNING NUMBERS'] = data['WINNING NUMBERS'].apply(lambda x: str(int(x)).zfill(3))

# Extract each individual digit from "WINNING NUMBERS" and put them into separate columns
data['winning_digit_1'] = data['WINNING NUMBERS'].str[0].astype("float64")
data['winning_digit_2'] = data['WINNING NUMBERS'].str[1].astype("float64")
data['winning_digit_3'] = data['WINNING NUMBERS'].str[2].astype("float64")

# Create lagged values of each number
for lag in range(1, 11):
    data[f"winning_number_lag_{lag}"] = data['WINNING NUMBERS'].shift(lag)

# Extract each individual digit from the lagged "WINNING NUMBERS" columns and put them into separate columns
for lag in range(1, 11):
    data[f"winning_digit_1_lag_{lag}"] = data[f"winning_number_lag_{lag}"].str[0].astype("float64")
    data[f"winning_digit_2_lag_{lag}"] = data[f"winning_number_lag_{lag}"].str[1].astype("float64")
    data[f"winning_digit_3_lag_{lag}"] = data[f"winning_number_lag_{lag}"].str[2].astype("float64")

# Convert lagged numbers back into float64
for lag in range(1, 11):
    data[f"winning_number_lag_{lag}"] = pd.to_numeric(data[f"winning_number_lag_{lag}"], errors='coerce').astype('float64')

# Convert the "WINNING NUMBERS" column to numerical format
data['WINNING NUMBERS'] = pd.to_numeric(data['WINNING NUMBERS'], errors='coerce').astype('float64')
data.dropna(subset=['WINNING NUMBERS'], inplace=True)

# Drop any missing values before modeling
data = data.dropna()

# Define the target column
target_col = 'WINNING NUMBERS'

print(data.dtypes)
print(data.shape)
print(data["WINNING NUMBERS"].shape)

# Get the features and target arrays
x = data.drop([target_col], axis=1).values
y = data[target_col].values

# Split the data into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

# Fit the StandardScaler to the training data
scaler.fit(x_train)

# Apply the transformation to both the training and testing data
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Initialize model project name by date
f = '%m-%d-%Y %H-%M-%S'
nw = datetime.datetime.now()
currentTime = nw.strftime(f)

# Initialize the AutoKeras model
clf = ak.TimeseriesForecaster(overwrite=False, max_trials=500, lookback=10, project_name=currentTime, directory='G:\\Other computers\\KUNA DESKTOP\\Creative Cloud Files\\python projs\\galottery\\models')

# Fit the model to the training data
clf.fit(x_train, y_train, epochs=2000, validation_data=(x_val, y_val))

# Make 5 predictions for the test data
n_samples = 5
predictions = []
for i in range(n_samples):
    prediction = clf.predict(x_val)
    predictions.append(prediction)

# Print the predictions
for i, prediction in enumerate(predictions):
    print("Prediction", i, ":", prediction)
