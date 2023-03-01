import pandas as pd
import autokeras as ak
import datetime
import sklearn.preprocessing as sk
import os
import tensorflow as tf
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import sys

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

# Iterate over each tuner, optimizer, regularization technique
for tuner in tuners:
    for optimizer in optimizers:
        for regularizer in regularizers:
            for name, scaler in scalers.items():

                # Specify model directory based on tuner, optimizer, regularization technique, and scaler
                model_dir = get_file_path("models beta6")
                os.makedirs(model_dir, exist_ok=True)
                
                # Set the project name
                project_name = f"{tuner}_{optimizer}_{regularizer}_{name}_{currentTime}"

                # Find the previous run
                latest_model_path = os.listdir(get_file_path("models beta6"))
                latest_model_path.sort(reverse=True)
                latest_model = None
                
                # If the latest model path contains something, set the latest model to the latest model path
                if len(latest_model_path) > 0:
                    latest_model = os.path.basename(latest_model_path[0])

                # If latest model is not none, change TensorBoard current, old directory, and callback to the latest model
                if latest_model is not None:
                    tensorboard_dir = os.path.join(get_file_path("tensorboard beta6"), latest_model)
                    old_tb_dir = os.path.join(get_file_path("old tb beta6"), latest_model)
                    # Redefine TensorBoard callback
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=os.path.join(get_file_path("tensorboard beta6"), latest_model), histogram_freq=50, 
                        write_graph=True, write_images=True, update_freq='batch', 
                        profile_batch=1, write_steps_per_second=False
                    )
                else:
                    # Specify TensorBoard directory based on tuner, optimizer, regularization technique, and scaler
                    tensorboard_dir = os.makedirs(os.path.join(get_file_path("tensorboard beta6"), project_name))
                    old_tb_dir = os.makedirs(os.path.join(get_file_path("old tb beta6"), project_name))

                # Define callbacks
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=tensorboard_dir, histogram_freq=50, 
                    write_graph=True, write_images=True, update_freq='batch', 
                    profile_batch=1, write_steps_per_second=False
                )
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=get_file_path("checkpoints beta6",
                    filename=(f"{name} {currentTime} checkpoint")), 
                    save_freq=44700, verbose=1,
                    save_best_only=True
                )
                stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=30
                )
                
                # Define callbacks list
                callbacks = [checkpoint_callback, tensorboard_callback, stopping_callback]

                # # Write a batch file to initialize the conda tf environment and TensorBoard in the same cmd window with a sleep after the environment is initialized
                # with open(get_file_path("start_tensorboard.bat"), "w") as f:
                #     f.write(f"tasklist | find \"chrome.exe\" && taskkill /im chrome.exe /f\n")
                #     f.write(f"conda activate tf\n")
                #     f.write(f"sleep 3\n")
                #     f.write(f"tensorboard --logdir='{tensorboard_dir}'\n")
                #     f.write(f"sleep 3\n")
                #     f.write(f"tasklist | find \"chrome.exe\" || start chrome http://localhost:6006/?darkMode=true#hparams\n")
                #     f.write(f"sleep 3\n")

                # # Run the batch file
                # os.system(get_file_path("start_tensorboard.bat"))

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
                
                # Fit the model
                clf.fit(x_train_scaled, y_train, validation_data=(x_val_scaled, y_val), epochs=None, shuffle=False, callbacks=callbacks)
                
                # Evaluate the model but if there is an error, clear the session and try again
                try:
                    print("Evaluating model...")
                    predictions = clf.predict(x_test_scaled)
                    error = np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)
                    print(f"Percentage error: {np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)}")
                except:
                    print("Error evaluating model, clearing session and trying again...")
                    tf.keras.backend.clear_session()
                    clf = run_model()
                    print("Evaluating model...")
                    predictions = clf.predict(x_test_scaled)
                    error = np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)
                    print(f"Percentage error: {np.mean((np.abs(y_test - predictions) / np.abs(predictions)) * 100)}")

                # Evaluate the trained model on the test set and print the results to a text file                
                with open(get_file_path("results beta6", filename="results.txt"), "a") as f:
                    f.write(f"{model_dir}, Percentage Error: {error}\n\n")

                # Put the name of the lowest error model at the top of the text file
                with open(get_file_path("results beta6", filename="results.txt"), "r") as f:
                    lines = f.readlines()
                with open(get_file_path("results beta6", filename="results.txt"), "w") as f:
                    lines.sort(key=lambda x: str(x.split(" ")[-1]))
                    f.writelines(lines)
                
                # Move the model to the "finished models" folder and move the TensorBoard model folder to "old tb files" folder if it exists
                shutil.move(get_file_path(model_dir, filename=project_name), get_file_path("finished models beta6"))
                if os.path.exists(tensorboard_dir):
                    shutil.move(tensorboard_dir, old_tb_dir)
                
                sys.exit()