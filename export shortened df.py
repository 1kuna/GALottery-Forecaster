import pandas as pd
import os

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

# Remove rows 2 to 10000
data = data.drop(data.index[1:11000])

# Drop any missing values before modeling
data = data.dropna()

# Export the data to a paraquet file
data.to_parquet(os.path.join(base_path, "fullcomb", "short.parquet").replace("/", "\\"))
print("Data exported to parquet file")

# Export data to a csv file
data.to_csv(os.path.join(base_path, "fullcomb", "short.csv").replace("/", "\\"), index=False)
print("Data exported to csv file")