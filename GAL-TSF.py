import pandas as pd
import autokeras as ak
import datetime
import sklearn.preprocessing as sk
import os

# Initialize scalers
robust = sk.RobustScaler()
minmax = sk.MinMaxScaler()
standard = sk.StandardScaler()

# Get the absolute path of the directory containing the script
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct the full file path to the combined csv subfolder, then get the file
file_path = os.path.join(base_path, "fullcomb", "combined backup.csv").replace("/", "\\")

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

# Fit the StandardScaler to the training data
scalers = {'robust': robust, 'minmax': minmax, 'standard': standard}

x_train_scaled = {}
x_val_scaled = {}
x_test_scaled = {}

for name, scaler in scalers.items():
    scaler.fit(x_train)
    x_train_scaled[name] = scaler.transform(x_train)
    x_val_scaled[name] = scaler.transform(x_val)
    x_test_scaled[name] = scaler.transform(x_test)

# Name current scaling method
current = 'minmax'

# Initialize model project name by date
f = '%m-%d-%Y %H-%M-%S'
nw = datetime.datetime.now()
currentTime = nw.strftime(f)

# Construct the full folder path to the "models" subfolder
model_save = os.path.join(base_path, "models").replace("/", "\\")

csv_save = os.path.join(base_path, "sheets", currentTime).replace("/", "\\")
os.makedirs(csv_save)

# Save each dataframe into a csv for later predictions
sets = [("train_scaled", x_train_scaled), ("val_scaled", x_val_scaled), ("test_scaled", x_test_scaled)]

for set_name, set_data in sets:
    set_df = pd.DataFrame(set_data[current])
    set_df.to_csv(f"{csv_save}\{set_name}.csv", index=False)

print(f"X Training Data Shape: {x_train_scaled[current].shape} | \
    X Validation Data Shape: {x_val_scaled[current].shape} | \
    X Test Data Shape: {x_test_scaled[current].shape} | \
    Y Training Data Shape: {y_train.shape} | \
    Y Validation Data Shape: {y_val.shape} | \
    Y Test Data Shape: {y_test.shape} | \
    ")

# Initialize the AutoKeras model
clf = ak.TimeseriesForecaster(overwrite=False, max_trials=2500, lookback=21, project_name=(f"beta3 {current} {currentTime}"), directory=model_save)

# Fit the model to the training data
clf.fit(x_train_scaled[current], y_train, epochs=None, validation_data=(x_val_scaled[current], y_val))

model = clf.export_model()

model.summary()

print(clf.evaluate(x_val_scaled[current], y_val))