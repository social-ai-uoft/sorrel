import argparse
import subprocess
import sys
from importlib import util
from pathlib import Path


def run_example(args, extra_args):
    example_name = args.example

    # Check if the example exists
    # We assume examples are in sorrel.examples.<name>
    # We can check by trying to find the module specification
    module_name = f"sorrel.examples.{example_name}.main"
    if not util.find_spec(module_name):
        print(
            f"Error: Example '{example_name}' not found or does not have a main module."
        )
        print(f"Expected module: {module_name}")
        return 1

    # Construct the command
    cmd = [sys.executable, "-m", module_name] + extra_args

    if args.config:
        cmd.append(f"--config-name={args.config}")

    try:
        # Run the command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running example '{example_name}': {e}")
        return e.returncode
    except KeyboardInterrupt:
        pass

    return 0


def show_logs(args, extra_args):
    example_name = args.example

    # Try to find the example module
    module_name = f"sorrel.examples.{example_name}"
    spec = util.find_spec(module_name)
    if not spec or not spec.submodule_search_locations:
        print(f"Error: Example '{example_name}' not found.")
        print(f"Expected module: {module_name}")
        return 1

    # Construct the path to the data directory
    example_dir = Path(spec.submodule_search_locations[0])
    data_dir = example_dir / "data"

    if not data_dir.exists():
        print(
            f"Warning: Data directory '{data_dir}' does not exist. TensorBoard might not have any data to show."
        )

    # Execute tensorboard
    # We use `tensorboard` command directly, which should be available in the environment
    cmd = ["tensorboard", "--logdir", str(data_dir)] + extra_args

    try:
        print(f"Starting TensorBoard for '{example_name}' at {data_dir}...")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running TensorBoard: {e}")
        return e.returncode
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(
            "Error: 'tensorboard' command not found. Please ensure it is installed and in your PATH."
        )
        return 1

    return 0


def gui(args, extra_args):
    cmd = ["streamlit", "run", "sorrel/utils/builder/Home.py"]
    try:
        print(f"Starting Streamlit GUI...")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit GUI: {e}")
        return e.returncode
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(
            "Error: 'streamlit' command not found. Please ensure it is installed and in your PATH."
        )
        return 1

    return 0


parser = argparse.ArgumentParser(
    description="Sorrel CLI tool. Use 'sorrel --help' for more information."
)
subparsers = parser.add_subparsers(dest="command", help="Available commands")

# 'run' subcommand
run_parser = subparsers.add_parser("run", help="Run a Sorrel example.")
run_parser.add_argument(
    "example", help="Name of the example to run (e.g., cleanup, chess)."
)
run_parser.add_argument(
    "--config",
    help="Name of the configuration file to use (without extension). Defaults to 'config'.",
)

# 'show-logs' subcommand
logs_parser = subparsers.add_parser(
    "show-logs",
    help="Start a TensorBoard session for a specific example's data logs",
)
logs_parser.add_argument("example", help="Name of the example (e.g., cleanup, chess).")

# 'gui' subcommand
gui_parser = subparsers.add_parser("gui", help="Start the Sorrel GUI.")


def main():
    # Parse known args to separate the command/example from the rest
    args, extra_args = parser.parse_known_args()

    if args.command == "run":
        sys.exit(run_example(args, extra_args))
    elif args.command == "show-logs":
        sys.exit(show_logs(args, extra_args))
    elif args.command == "gui":
        sys.exit(gui(args, extra_args))
    else:
        parser.print_help()


if __name__ == "__main__":

    main()
