# Setup
# Step 1: Install poetry globally via pipx (run once)
# pipx install poetry
# Step 2: Create a clean conda environment (run every time) (make sure that you are in PS D:\Coding\Sorrel\Code>)
# conda create --name sorrel_env python=3.12
# Step 3: Initialize conda (run every time)
# conda init powershell
# Step 4: Activate the conda environment (run every time)
# conda activate sorrel_env
# Step 5: Tell poetry to use the current environment (run every time)
# poetry config virtualenvs.create false
# Step 6: Install Sorrel from the repo directory (run every time)
# poetry install
# Step 7: (this example is located in D:\Coding\Sorrel\Code\sorrel\examples\treasurehunt)
# poetry run python -m sorrel.examples.treasurehunt.main
# Step 8: Close the environment (run every time)
# conda deactivate

# Important notes:
# Use python -m pipx <command here>

#import numpy as np
#print(np.linalg.norm([1,2,3]))