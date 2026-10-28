"""Command-line interface. Help and comparison never load IMDB."""

import argparse
from .config import load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="IMDB: TF-IDF Logistic Regression / SVM / RNN / LSTM")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "smoke", "compare"):
        child = commands.add_parser(command)
        child.add_argument("--config", default="configs/default.yaml")
        child.add_argument("--run-id", help="Append missing models to this run, or select it for comparison")
        if command == "run":
            child.add_argument("--model", choices=["all", "logreg", "svm", "rnn", "lstm"], default="all")
    args = parser.parse_args(argv)
    try:
        mode = "smoke" if args.command == "smoke" else "full"
        config = load_config(args.config, mode)
        from .reporting import compare_results, find_run
        if args.command == "compare":
            table = compare_results(find_run(config, args.run_id))
        else:
            from .data import load_data
            from .experiment import Experiment, MODEL_NAMES
            bundle = load_data(config, mode)
            experiment = Experiment(config, bundle, mode, args.run_id)
            names = MODEL_NAMES if args.command == "smoke" or args.model == "all" else (args.model,)
            for name in names:
                experiment.train(name)
            table = compare_results(experiment.directory) if len(names) == 4 else None
        if table is not None:
            print(table.to_string(index=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
