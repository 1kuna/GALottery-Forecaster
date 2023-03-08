import pickle
import os

current_tuner_index = 0
current_optimizer_index = 0
current_scaler_index = 0

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

# Load the loop state if the program is interrupted
def load_pickle():
    with open(get_file_path("loop_state.pickle"), "rb") as f:
        state = pickle.load(f)
        current_tuner_index, current_optimizer_index, current_scaler_index = state
        print(f"Resuming from: (tuner: {current_tuner_index}, optimizer: {current_optimizer_index}, and scaler: {current_scaler_index})")

# Create Pickle save function
def save_pickle():
    with open(get_file_path("loop_state.pickle"), "wb") as f:
        pickle.dump((current_tuner_index, current_optimizer_index, current_scaler_index), f)
        print(f"Saved state: (tuner: {current_tuner_index}, optimizer: {current_optimizer_index}, and scaler: {current_scaler_index})")

load_pickle()

# Change values here
# current_tuner_index = 0
# current_optimizer_index = 1
# current_scaler_index = 0

save_pickle()