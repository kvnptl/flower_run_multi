import yaml
import subprocess
import os
from datetime import datetime
import argparse


def run_experiments(config_file="experiments_config.yaml"):
    # Load experiment configurations from YAML
    with open(config_file, "r") as f:
        experiments = yaml.safe_load(f)["experiments"]

    # Run each experiment
    for exp in experiments:
        # Construct the `--run-config` string from parameters
        run_config = f"num-server-rounds={exp['num-server-rounds']} "
        run_config += f"fraction-fit={exp['fraction-fit']} "
        run_config += f"local-epochs={exp['local-epochs']}"

        # Generate a unique log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"output_{exp['name']}_{timestamp}.log"

        print(f"Running {exp['name']}... (Logging to {log_filename})")

        # Run the Flower CLI command and log output
        with open(log_filename, "w") as log_file:
            process = subprocess.Popen(
                ["flwr", "run", ".", "larger-sim", "--run-config", run_config],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            process.wait()  # Wait for the process to finish

        print(
            f"Experiment {exp['name']} completed. Logs saved to {log_filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flower experiments.")
    parser.add_argument(
        "--config-file", default="experiments_config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_experiments(config_file=args.config_file)
