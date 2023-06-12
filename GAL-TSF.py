### HIGH PRIORITY
### TODO: fix error `tensorflow.python.framework.errors_impl.PermissionDeniedError: Failed to remove a directory: k:\Git\KUNA\GALottery-Forecaster\forecast\checkpoints\None_None_mse_minmax\chief; Permission denied`
### HIGH PRIORITY

# TODO: test cross platform compatibility; run an extended test on Mac to ensure compatibility
# TODO: run unit test for prediction result and text file output
# TODO: refactor and consolidate loop, potentially breaking it up into multiple functions/files
# TODO: create yaml file for configuration depenedent on the user's system, run this check at the start of the batch file
    # TODO: config includes setting up the conda environment and installing packages
# TODO: clean up file directory structure to be more direct and concise
# TODO: potentially just turn it into a single executable file with a ui showing progress
# TODO: figure out how to remove "val_loss metric unavailable" warning verbosity
# TODO: ignore cupti error

import pandas as pd
import autokeras as ak
import datetime
import sklearn.preprocessing as sk
import os
import tensorflow as tf
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
# import wmi
import sys
import platform

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the VRAM limit
vram_limit = None

# Limit the VRAM TensorFlow can use
gpus = tf.config.experimental.list_physical_devices('GPU')
if vram_limit is not None:
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vram_limit)])
            print(f"VRAM limit set to {vram_limit / 1024} GB")
        except RuntimeError as e:
            print(e)

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    if platform.system() == "Windows":
        full_path = full_path.replace("/", "\\")
    return full_path

# Read in the paraquet file
data = pd.read_parquet(get_file_path("fullcomb", filename="shortnew.parquet"))

# Define the target column
target_col = 'WINNING NUMBERS'

# Get the features and target arrays
x = data.drop([target_col], axis=1).values
y = data[target_col].values

# Split the data into training, validation, and test sets (80% for training, 10% for validation, and 10% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, shuffle=False)

# Initialize tuners, optimizers, loss functions, and scalers
tuners = ['random', 'bayesian', 'hyperband', 'greedy']
optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']
loss_funcs = ['mse', 'mae', 'msle', 'mape', 'huber_loss', 'log_cosh', 'poisson', 'cosine_similarity', 'log_cosh']
scalers = {'minmax': sk.MinMaxScaler(), 'standard': sk.StandardScaler(), 'robust': sk.RobustScaler()}

# Keep track of the progress within the for loop and resume from the last run if the program is interrupted
current_tuner_index = 0
current_optimizer_index = 0
current_loss_index = 0
current_scaler_index = 0

# Load the loop state if the program is interrupted
def load_pickle():
    with open(get_file_path("forecast/loop_state.pickle"), "rb") as f:
        state = pickle.load(f)
        current_tuner_index, current_optimizer_index, current_loss_index, current_scaler_index = state
        print(f"Resuming from: (tuner: {current_tuner_index}, optimizer: {current_optimizer_index}, loss function: {current_loss_index}, and scaler: {current_scaler_index})")
    return current_tuner_index, current_optimizer_index, current_loss_index, current_scaler_index

# Create Pickle save function
def save_pickle():
    with open(get_file_path("forecast/loop_state.pickle"), "wb") as f:
        pickle.dump((current_tuner_index, current_optimizer_index, current_loss_index, current_scaler_index), f)
        print(f"Saved state: (tuner: {current_tuner_index}, optimizer: {current_optimizer_index}, loss function: {current_loss_index}, and scaler: {current_scaler_index})")

# Load model before starting the loop
try:
    current_tuner_index, current_optimizer_index, current_loss_index, current_scaler_index = load_pickle()
except:
    save_pickle()

# Iterate over each tuner, optimizer, loss functions, and scaler
for tuner in tuners[current_tuner_index:]:
    for optimizer in optimizers[current_optimizer_index:]:
        for loss_func in loss_funcs[current_loss_index:]:
            for name, scaler in list(scalers.items())[current_scaler_index:]:

                # Load the loop state after each iteration
                current_tuner_index, current_optimizer_index, current_loss_index, current_scaler_index = load_pickle()

                # Print current tuner, optimizer, loss functions, and scaler
                print(f"Current tuner: {tuner}")
                print(f"Current optimizer: {optimizer}")
                print(f"Current loss function: {loss_func}")
                print(f"Current scaler: {name}")

                # Specify model directory and project name based on tuner, optimizer, loss function, and scaler
                model_dir = get_file_path("forecast\\models")
                project_name = f"{tuner}_{optimizer}_{loss_func}_{name}_{currentTime}"
                checkpoint_name = f"{tuner}_{optimizer}_{loss_func}_{name}"

                # Find the previous run
                latest_model_path = os.listdir(get_file_path("forecast/models"))
                latest_model_path.sort(reverse=True)
                latest_model = None
                
                # If the latest model path contains something, set the latest model to the latest model path
                if len(latest_model_path) > 0:
                    latest_model = os.path.basename(latest_model_path[0])

                # If latest model is not none, change TensorBoard current, old directory, and callback to the latest model
                if latest_model is not None:
                    tensorboard_dir = os.path.join(get_file_path("forecast/tensorboard"), latest_model)
                    old_tb_dir = os.path.join(get_file_path("forecast/old tb"), latest_model)
                    # Redefine TensorBoard callback
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=os.path.join(get_file_path("forecast/tensorboard"), latest_model), histogram_freq=50, 
                        write_graph=True, write_images=True, update_freq='batch', 
                        profile_batch=1, write_steps_per_second=False
                    )
                else:
                    # Specify TensorBoard directory based on tuner, optimizer, loss function, and scaler
                    tensorboard_dir = os.path.join(get_file_path("forecast/tensorboard"), project_name)
                    old_tb_dir = os.path.join(get_file_path("forecast/old tb"), project_name)


                # Define callbacks
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=tensorboard_dir, histogram_freq=50, 
                    write_graph=True, write_images=True, update_freq='batch', 
                    write_steps_per_second=False
                )

                checkpointing = tf.keras.callbacks.BackupAndRestore(
                    backup_dir=get_file_path(get_file_path("forecast/checkpoints", checkpoint_name)),
                    save_freq='epoch'
                )

                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000000, verbose=1, mode='min')

                # Define callbacks list
                callbacks = [checkpointing, early_stopping]

                # # Write a batch file to initialize the conda tf environment and TensorBoard in the same cmd window with a sleep after the environment is initialized
                # c = wmi.WMI()
                # # Write a batch file to initialize the conda tf environment and TensorBoard in the same cmd window with a sleep after the environment is initialized
                # with open(get_file_path("start_tensorboard.bat"), "w") as f:
                #     f.write(f"conda activate tf\n")
                #     f.write(f"sleep 3\n")
                #     f.write(f"tensorboard --logdir='{tensorboard_dir}'\n")
                #     f.write(f"sleep 3\n")
                #     # Start Chrome if it's not already running
                #     processes = c.Win32_Process(name='chrome.exe')
                #     if len(processes) == 0:
                #         f.write(f'start "" "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe" http://localhost:6006/?darkMode=true#hparams\n')
                #     f.write(f"sleep 3\n")

                # # Run the batch file
                # batch_file = get_file_path("start_tensorboard.bat").replace("\\", "\\\\")
                # os.system('"' + batch_file + '"')

                # Initialize the model
                # *** Remember that lookback needs to be a multiple of the batch size. Make this a solved issue on the repo ***
                def run_model():
                    clf = ak.TimeseriesForecaster(
                        tuner=tuner,
                        optimizer=optimizer,
                        max_trials=250,
                        lookback=320,
                        project_name=project_name,
                        directory=model_dir,
                        overwrite=False,
                        loss=loss_func,
                        metrics="mape"
                    )
                    return clf

                # Load model checkpoint if it exists, otherwise initialize new model
                if latest_model is not None:
                    print(f"Loading previous model: {latest_model}")
                    project_name = latest_model
                    clf = run_model()
                    print("Previous training run resumed successfully.")
                else:
                    print("No previous model found, initializing new model.")
                    clf = run_model()

                # Scale the data
                x_train_scaled = scaler.fit_transform(x_train)
                x_val_scaled = scaler.transform(x_val)
                x_test_scaled = scaler.transform(x_test)

                # Train the AutoKeras model
                clf.fit(x_train_scaled, y_train, validation_data=(x_val_scaled, y_val), epochs=None, shuffle=False, callbacks=callbacks, batch_size=64)

                # Evaluate the model but if there is an error, clear the session and try again
                try:
                    print("Evaluating model...")
                    predictions = clf.predict(x_test_scaled)
                    error = np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)
                    print(f"Percentage error: {error:.2f}")
                except:
                    print("Error evaluating model, clearing session and trying again...")
                    tf.keras.backend.clear_session()
                    clf = run_model()
                    print("Evaluating model...")
                    predictions = clf.predict(x_test_scaled)
                    error = np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)
                    print(f"Percentage error: {error:.2f}")
                                
                # Write the model name and error result to a text file and sort the file by error
                with open(get_file_path("forecast", filename="results.txt"), "a") as f:
                    f.write(f"Model: {project_name} || Percentage Error: {error:.2f}%\n\n")
                with open(get_file_path("forecast", filename="results.txt"), "r") as f:
                    lines = f.readlines()
                with open(get_file_path("forecast", filename="results.txt"), "w") as f:
                    if len(lines) > 3:
                        for line in sorted(lines, key=lambda x: float(x.split(": ")[1]) if len(x.split(": ")) > 1 else 0, reverse=True):
                            f.write(f"{line}\n")
                
                # Move the model to the "finished models" folder and move the TensorBoard model folder to "old tb files" folder if it exists
                shutil.move(get_file_path(model_dir, filename=project_name), get_file_path("forecast\\finished models"))
                if os.path.exists(tensorboard_dir):
                    shutil.move(tensorboard_dir, old_tb_dir)

                # Clear the session
                tf.keras.backend.clear_session()

                # Increment current scaler index
                if current_scaler_index < len(scalers) - 1:
                    current_scaler_index += 1
                    save_pickle()
                else:
                    current_scaler_index = 0
                    save_pickle()

            # Increment current loss index
            if current_loss_index < len(loss_funcs) - 1:
                current_loss_index += 1
                save_pickle()
            else:
                current_loss_index = 0
                save_pickle()

        # Increment current optimizer index
        if current_optimizer_index < len(optimizers) - 1:
            current_optimizer_index += 1
            save_pickle()
        else:
            current_scaler_index = 0
            save_pickle()
                
    # Increment current tuner index
    if current_tuner_index < len(tuners) - 1:
        current_tuner_index += 1
        save_pickle()
    else:
        current_tuner_index = 0
        save_pickle()