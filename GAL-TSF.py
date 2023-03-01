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
import wmi
import sys

# Set TensorFlow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize current time
currentTime = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')

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

# Split the data into training, validation, and test sets (80% for training, 10% for validation, and 10% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, shuffle=False)

# Initialize tuners, optimizers, regularization techniques, and scalers
tuners = ['random', 'bayesian', 'hyperband', 'greedy']
optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']
regularizers = [None, 'l1', 'l2']
scalers = {'robust': sk.RobustScaler(), 'minmax': sk.MinMaxScaler(), 'standard': sk.StandardScaler()}

# Keep track of the progress within the for loop and resume from the last run if the program is interrupted
current_tuner_index = 0
current_optimizer_index = 0
current_regularizer_index = 0
current_scaler_index = 0

# Load the loop state if the program is interrupted
with open(get_file_path("loop_state.pickle"), "rb") as f:
    state = pickle.load(f)
    current_tuner_index, current_optimizer_index, current_regularizer_index, current_scaler_index = state
    print(f"Resuming from: (tuner: {current_tuner_index}, optimizer: {current_optimizer_index}, regularizer: {current_regularizer_index}, and scaler: {current_scaler_index})")

# Iterate over each tuner, optimizer, regularization technique, and scaler
for tuner in tuners[current_tuner_index:]:
    for optimizer in optimizers[current_optimizer_index:]:
        for regularizer in regularizers[current_regularizer_index:]:
            for name, scaler in list(scalers.items())[current_scaler_index:]:

                # Save the loop state if the program is interrupted
                with open(get_file_path("loop_state.pickle"), "wb") as f:
                    pickle.dump((current_tuner_index, current_optimizer_index, current_regularizer_index, current_scaler_index), f)

                # Print current tuner, optimizer, regularization technique, and scaler
                print(f"Current tuner: {tuner}")
                print(f"Current optimizer: {optimizer}")
                print(f"Current regularizer: {regularizer}")
                print(f"Current scaler: {name}")

                # Specify model directory and project name based on tuner, optimizer, regularization technique, and scaler
                model_dir = get_file_path("forecast\\models")
                project_name = f"{tuner}_{optimizer}_{regularizer}_{name}_{currentTime}"            

                # Find the previous run
                latest_model_path = os.listdir(get_file_path("forecast\\models"))
                latest_model_path.sort(reverse=True)
                latest_model = None
                
                # If the latest model path contains something, set the latest model to the latest model path
                if len(latest_model_path) > 0:
                    latest_model = os.path.basename(latest_model_path[0])

                # If latest model is not none, change TensorBoard current, old directory, and callback to the latest model
                if latest_model is not None:
                    tensorboard_dir = os.path.join(get_file_path("forecast\\tensorboard"), latest_model)
                    old_tb_dir = os.path.join(get_file_path("forecast\\old tb"), latest_model)
                    # Redefine TensorBoard callback
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=os.path.join(get_file_path("forecast\\tensorboard"), latest_model), histogram_freq=50, 
                        write_graph=True, write_images=True, update_freq='batch', 
                        profile_batch=1, write_steps_per_second=False
                    )
                else:
                    # Specify TensorBoard directory based on tuner, optimizer, regularization technique, and scaler
                    tensorboard_dir = os.path.join(get_file_path("forecast\\tensorboard"), project_name)
                    old_tb_dir = os.path.join(get_file_path("forecast\\old tb"), project_name)


                # Define callbacks
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=tensorboard_dir, histogram_freq=50, 
                    write_graph=True, write_images=True, update_freq='batch', 
                    write_steps_per_second=False
                )
                stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=30
                )
                
                # Define callbacks list
                callbacks = [tensorboard_callback, stopping_callback]

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
                def run_model():
                    clf = ak.TimeseriesForecaster(
                        tuner=tuner,
                        optimizer=optimizer,
                        max_trials=250,
                        lookback=21,
                        project_name=project_name,
                        directory=model_dir,
                        overwrite=False,
                        objective='val_loss'
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

                # Apply L1/L2 regularization
                if regularizer == 'l1':
                    clf.add_regularizer(tf.keras.regularizers.l1(0.01))
                elif regularizer == 'l2':
                    clf.add_regularizer(tf.keras.regularizers.l2(0.01))

                # Scale the data
                x_train_scaled = scaler.fit_transform(x_train)
                x_val_scaled = scaler.transform(x_val)
                x_test_scaled = scaler.transform(x_test)

                # Train the AutoKeras model
                clf.fit(x_train_scaled, y_train, validation_data=(x_val_scaled, y_val), epochs=None, shuffle=False, callbacks=callbacks)

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

                # Evaluate the trained model on the test set and print the results to a text file                
                with open(get_file_path("forecast\\results", filename="results.txt"), "a") as f:
                    f.write(f"{model_dir}, Percentage Error: {error}\n\n")

                # Write the model name and error result to a text file and sort the file by error
                with open(get_file_path("forecast\\results", filename="results.txt"), "a") as f:
                    f.write(f"{model_dir}, Percentage Error: {error}\n\n")
                with open(get_file_path("forecast\\results", filename="results.txt"), "r") as f:
                    lines = f.readlines()
                with open(get_file_path("forecast\\results", filename="results.txt"), "w") as f:
                    if len(lines) > 3:
                        for line in sorted(lines, key=lambda x: float(x.split(": ")[1]) if len(x.split(": ")) > 1 else 0, reverse=True):
                            f.write(line)
                
                # Move the model to the "finished models" folder and move the TensorBoard model folder to "old tb files" folder if it exists
                shutil.move(get_file_path(model_dir, filename=project_name), get_file_path("forecast\\finished models"))
                if os.path.exists(tensorboard_dir):
                    shutil.move(tensorboard_dir, old_tb_dir)

                # Clear the session
                tf.keras.backend.clear_session()

                # Increment current scaler index
                current_scaler_index += 1
            # Increment current regularizer index
            current_regularizer_index += 1
        # Increment current optimizer index
        current_optimizer_index += 1
    # Increment current tuner index
    current_tuner_index += 1