#!/bin/bash

# Specify the Python script and config file
SCRIPT="run_experiments.py"
CONFIG_FILE="experiments_config.yaml"

# Ensure the script is executable
chmod +x $SCRIPT

# Run the Python script and pass the config file
python $SCRIPT --config-file $CONFIG_FILE
