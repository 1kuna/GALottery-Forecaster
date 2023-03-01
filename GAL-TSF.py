import pandas as pd
import autokeras as ak
import datetime
import sklearn.preprocessing as sk
import os
import tensorflow as tf
import shutil
from sklearn.model_selection import train_test_split

# Name current scaling method
current = 'robust'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Read in the paraquet file
data = pd.read_parquet(get_file_path("fullcomb", filename="short.parquet"))

# Define the target column
target_col = 'WINNING NUMBERS'

# Get the features and target arrays
x = data.drop([target_col], axis=1).values
y = data[target_col].values

# Split the data into training, validation, and test sets (70% for training, 15% for validation, and 15% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, shuffle=False)

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

# Initialize model project name by date
f = '%m-%d-%Y %H-%M-%S'
nw = datetime.datetime.now()
currentTime = nw.strftime(f)

# Define directories and create if necessary
model_dir = get_file_path("models")
os.makedirs(model_dir, exist_ok=True)

sheet_dir = get_file_path("sheets", currentTime)
os.makedirs(sheet_dir, exist_ok=True)

# Save each dataframe into a csv for later predictions
sets = [("train_scaled", x_train_scaled), ("val_scaled", x_val_scaled), ("test_scaled", x_test_scaled)]

for set_name, set_data in sets:
    set_df = pd.DataFrame(set_data[current])
    set_df.to_csv(get_file_path(sheet_dir, filename=f"{set_name}.csv"), index=False, header=True)

# Initialize the AutoKeras model
clf = ak.TimeseriesForecaster(
    overwrite=False, max_trials=2, lookback=21, 
    project_name=(f"beta5 {current} {currentTime}"), 
    directory=model_dir
)

# # Remove previous tensorboard logs if they exist
# tensorboard_dir = get_file_path("tensorboard")
# tensorboard_subdirs = [d for d in os.listdir(tensorboard_dir) if os.path.isdir(os.path.join(tensorboard_dir, d))]
# for subdir in tensorboard_subdirs:
#     shutil.rmtree(os.path.join(tensorboard_dir, subdir))

# Open Tensorboard in browser
# os.system(f"start cmd /k tensorboard --logdir={tensorboard_dir}")

# Define callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=get_file_path("tensorboard"), histogram_freq=50, 
    write_graph=True, write_images=True, update_freq='batch', 
    profile_batch=1, write_steps_per_second=False
)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=get_file_path("checkpoints",
    filename=(f"{current} {currentTime} checkpoint")), 
    save_freq=89400, verbose=1
)
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
)

callbacks = [checkpoint_callback, tensorboard_callback, stopping_callback]

# Fit, summarize, evaluate, and export the model
clf.fit(x_train_scaled[current], y_train, epochs=25, validation_data=(x_val_scaled[current], y_val), shuffle=False)
print(clf.predict(x_test_scaled[current]))
model = clf.export_model()
print(model.summary())
