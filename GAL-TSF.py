import pandas as pd
import autokeras as ak
import datetime
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as sk
import os

# Initialize scalers
robust = sk.RobustScaler()
minmax = sk.MinMaxScaler()
standard = sk.StandardScaler()

# Get the absolute path of the directory containing the script
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct the full file path to the combined csv subfolder, then get the file
file_path = os.path.join(base_path, "fullcomb", "combined.csv").replace("/", "\\")

# Read in the csv file
data = pd.read_csv(file_path, header=0)

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

# Split the data into training, validation, and testing sets (not randomized)
split_1 = int(x.shape[0] * 0.7)
split_2 = int(x.shape[0] * 0.85)
x_train = x[:split_1]
y_train = y[:split_1]
x_val = x[split_1:split_2]
y_val = y[split_1:split_2]
x_test = x[split_2:]
y_test = y[split_2:]

x_train_1 = x_train
x_train_2 = x_train
x_train_3 = x_train
x_test_1 = x_test
x_test_2 = x_test
x_test_3 = x_test

# Fit the StandardScaler to the training data
robust.fit(x_train)
minmax.fit(x_train)
standard.fit(x_train)

x_train_minmax = x_train_1
x_train_robust = x_train_2
x_train_standard = x_train_3
x_test_robust = x_test_2
x_test_standard = x_test_3
x_test_minmax = x_test_1

# Apply the transformation to both the training and testing data
x_train_robust = robust.transform(x_train_robust)
x_train_standard = standard.transform(x_train_standard)
x_train_minmax = minmax.transform(x_train_minmax)
x_val_robust = robust.transform(x_val)
x_test_robust = robust.transform(x_test)
x_val_minmax = minmax.transform(x_val)
x_test_minmax = minmax.transform(x_test)
x_val_standard = standard.transform(x_val)
x_test_standard = standard.transform(x_test)
x_test_robust = robust.transform(x_test_robust)
x_test_standard = standard.transform(x_test_standard)
x_test_minmax = minmax.transform(x_test_minmax)

# Initialize model project name by date
f = '%m-%d-%Y %H-%M-%S'
nw = datetime.datetime.now()
currentTime = nw.strftime(f)

# Construct the full folder path to the "models" subfolder
model_save = os.path.join(base_path, "models").replace("/", "\\")

# Initialize the AutoKeras model
clf = ak.TimeseriesForecaster(overwrite=False, max_trials=5, lookback=15, project_name=(f"beta2 standard {currentTime}"), directory=model_save)

# Fit the model to the training data
clf.fit(x_train_standard, y_train, epochs=20, validation_data=(x_val_standard, y_val))

model = clf.export_model()

model.summary()

# # Make 5 predictions for the test data
# n_samples = 5
# predictions = []
# for i in range(n_samples):
#     prediction = clf.predict(x_test_standard)
#     print(predictions.shape)
#     print(clf.evaluate(x_val_standard, y_val))
#     predictions.append(prediction)

# # Print the predictions
# for i, prediction in enumerate(predictions):
#     print("Prediction", i, ":", prediction)
