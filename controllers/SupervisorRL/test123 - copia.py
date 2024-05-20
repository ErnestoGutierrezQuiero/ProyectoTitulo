import pickle
import numpy as np
# Path to the pickle file
pickle_file_path = "best_actions.pkl"

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    best_action_data = pickle.load(file)

# Explore the data
print("Type of data:", type(best_action_data))

# If it's a dictionary, list the keys
if isinstance(best_action_data, dict):
    print("Keys:", best_action_data.keys())

# Print some values or specific parts of the data
# This step depends on the data structure and your goal
# For example, you might want to print the entire content or specific values
print("Sample data:", best_action_data)

# Additional exploration, depending on data type
if isinstance(best_action_data, dict):
    for key, value in best_action_data.items():
        print(f"State {key}: Best action(s) - {value}")

# Or if it's a list/array, you can explore its content similarly
print("-----------------------------------------------------------------------------------------")
pickle_file_path = "q_table.pkl"

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    q_table = pickle.load(file)



best_actions = np.zeros((q_table.shape[0], q_table.shape[1]), dtype=int)
for i in range(q_table.shape[0]):
    for j in range(q_table.shape[1]):
        best_actions[i, j] = np.argmax(q_table[i, j, :])

# Display best actions
print("Best actions for each state:")
print(best_actions)

# Optional: Create a human-readable interpretation of the actions
action_names = ["Stop", "Move Forward", "Turn Right", "Turn Left", "Move Backward"]

human_readable_best_actions = np.vectorize(lambda x: action_names[x])(best_actions)

print("Human-readable best actions:")
print(human_readable_best_actions)