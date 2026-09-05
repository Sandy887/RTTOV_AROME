"""
A utility script to manage a simple counter stored in a text file (`count.txt`) and log experiment details.

Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com

Components:
    - `read_count()`: Reads the current count from `count.txt`.
    - `write_count(count)`: Writes the updated count to `count.txt`.
    - `main()`: Demonstrates how to read, update, and write the count.
    - `log_experiment(details)`: Appends experiment details to `experiments.txt`, one per line.

Usage:
    - Tracks and updates a numeric count between runs.
    - Logs metadata or comments related to each experiment run.
"""

import os

# Function to read count from the file
def read_count(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                return int(file.read().strip())
            except ValueError:
                # Handle file corruption or invalid data
                print("Invalid count in file, resetting to 0.")
                return 0
    else:
        return 0

# Function to write count to the file
def write_count(file_path, count):
    with open(file_path, 'w') as file:
        file.write(str(count))

# Main function to demonstrate updating the count
def update_count(file_path, increment=1):
    count = read_count(file_path)  # Read the current count
    count += increment             # Update the count
    write_count(file_path, count)  # Save the updated count
    #print(f"Updated count: {count}")
    return count

def log_experiment(exp_name, exp_purpose,exp_date, file_path):
    print(f"Save the experiment name and purpose inside {file_path[-30:]}")
    with open(file_path, 'a') as file:
        # Write the experiment details, each on a new line
        file.write(f"{exp_name}\t{exp_purpose}\t{exp_date}\n")
