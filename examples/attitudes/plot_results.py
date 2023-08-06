import glob
import numpy as np


def process_files(file_pattern="rs*p2.txt", smoothing=1):
    # Dictionary to store data
    data_dict = {}

    # Iterate over each file matching the pattern
    for filename in glob.glob(file_pattern):
        with open(filename, "r") as file:
            for line in file:
                # Check if line starts with a number
                if line[0].isdigit():
                    parts = line.split()
                    time_step = int(parts[0])
                    third_integer = int(parts[2])
                    last_string = parts[-1]

                    # Add data to dictionary
                    if last_string not in data_dict:
                        data_dict[last_string] = {}
                    if time_step not in data_dict[last_string]:
                        data_dict[last_string][time_step] = []
                    data_dict[last_string][time_step].append(third_integer)

    # Function for smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Process and print results
    for key, value in data_dict.items():
        print(f"Results for {key}:")
        time_steps = sorted(value.keys())
        averages = []
        stderrs = []
        for t in time_steps:
            mean = np.mean(value[t])
            stderr = np.std(value[t]) / np.sqrt(len(value[t]))
            averages.append(mean)
            stderrs.append(stderr)

        # Apply smoothing if applicable
        if smoothing > 1:
            averages = smooth(averages, smoothing)
            stderrs = smooth(stderrs, smoothing)

        for t, avg, err in zip(time_steps[: len(averages)], averages, stderrs):
            print(f"Time step: {t}, Average: {avg}, Standard Error: {err}")

        print()


# Example usage
process_files(smoothing=5)
