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


def main():
    parser = argparse.ArgumentParser(description="Sorrel CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run a Sorrel example")
    run_parser.add_argument(
        "example", help="Name of the example to run (e.g., cleanup, chess)"
    )
    run_parser.add_argument(
        "--config",
        help="Name of the configuration file to use (without extension). Defaults to 'config'.",
    )

    # Parse known args to separate the command/example from the rest
    args, extra_args = parser.parse_known_args()

    if args.command == "run":
        sys.exit(run_example(args, extra_args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
