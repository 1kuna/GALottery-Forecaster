import pandas as pd
import autokeras as ak
import datetime
import sklearn.preprocessing as sk
import os
import tensorflow as tf
import shutil
from sklearn.model_selection import train_test_split

# Get the absolute path of the directory containing the script
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct the full file path to the combined csv subfolder, then get the file
file_path = os.path.join(base_path, "fullcomb", "combined.parquet").replace("/", "\\")

# Read in the paraquet file
data = pd.read_parquet(file_path)

# Define the target column
target_col = 'WINNING NUMBERS'

# Get the features and target arrays
x = data.drop([target_col], axis=1).values
y = data[target_col].values

# Split the data into training, validation, and test sets (70% for training, 15% for validation, and 15% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=False)

# Initialize scalers
scalers = {
    'robust': sk.RobustScaler(),
    'minmax': sk.MinMaxScaler(),
    'standard': sk.StandardScaler()
}

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

# Initialize the AutoKeras model
clf = ak.TimeseriesForecaster(overwrite=False, max_trials=2500, lookback=21, project_name=(f"beta4 {current} {currentTime}"), directory=model_save)

# Remove previous tensorboard logs if there are more than 15 folders
tensorboard_dir = f"{base_path}\\tensorboard"
tensorboard_subdirs = [d for d in os.listdir(tensorboard_dir) if os.path.isdir(os.path.join(tensorboard_dir, d))]
if len(tensorboard_subdirs) > 15:
    oldest_subdir = sorted(tensorboard_subdirs)[0]
    shutil.rmtree(os.path.join(tensorboard_dir, oldest_subdir))

# Define callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f"{base_path}\\tensorboard", 
    histogram_freq=30, 
    write_graph=True,
    write_images=True, 
    update_freq='batch', 
    profile_batch=2
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{base_path}\\checkpoints\\{current} {currentTime}", 
    save_freq=1
)

callbacks = [tensorboard_callback, checkpoint_callback]

# Fit the model to the training data
clf.fit(x_train_scaled[current], y_train, epochs=None, validation_data=(x_val_scaled[current], y_val), callbacks=callbacks)

model = clf.export_model()

model.summary()

print(clf.evaluate(x_val_scaled[current], y_val))