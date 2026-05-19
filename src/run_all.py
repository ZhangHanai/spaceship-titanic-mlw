"""Run one or more Spaceship Titanic model training scripts safely.

Examples from the repository root:
    python src/run_all.py --models svm random_forest logistic_regression
    python src/run_all.py --models lightgbm
    python src/run_all.py --skip-heavy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from utils import REPO_ROOT


@dataclass(frozen=True)
class ModelCommand:
    """Command metadata for a runnable model script."""

    name: str
    script: Path
    heavy: bool = False


MODEL_COMMANDS: dict[str, ModelCommand] = {
    "svm": ModelCommand("svm", REPO_ROOT / "src" / "train_svm.py"),
    "random_forest": ModelCommand("random_forest", REPO_ROOT / "src" / "train_random_forest.py"),
    "logistic_regression": ModelCommand(
        "logistic_regression",
        REPO_ROOT / "src" / "train_logistic_regression.py",
    ),
    "lightgbm": ModelCommand("lightgbm", REPO_ROOT / "src" / "train_lightgbm.py", heavy=True),
    "xgboost": ModelCommand("xgboost", REPO_ROOT / "src" / "train_xgboost.py", heavy=True),
}

DEFAULT_MODELS = ["svm", "random_forest", "logistic_regression", "lightgbm", "xgboost"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run selected Spaceship Titanic model scripts.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_COMMANDS),
        default=DEFAULT_MODELS,
        help="Model scripts to run. Defaults to all available models.",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip heavier models and use fast options where available for demo smoke runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue running later models if one script fails. This is the default.",
    )
    return parser.parse_args()


def command_for_model(model: ModelCommand, skip_heavy: bool) -> list[str]:
    """Build the subprocess command for one model."""
    command = [sys.executable, str(model.script)]
    if skip_heavy and model.name == "svm":
        command.append("--skip-tuning")
    if skip_heavy and model.name == "logistic_regression":
        command.extend(["--optuna-trials", "5"])
    return command


def main() -> int:
    """Run selected models and print a success/failure summary."""
    args = parse_args()
    requested_models = list(dict.fromkeys(args.models))
    if args.skip_heavy:
        requested_models = [name for name in requested_models if not MODEL_COMMANDS[name].heavy]

    if not requested_models:
        print("No models selected after applying --skip-heavy.")
        return 0

    results: list[dict[str, object]] = []
    print("Repository root:", REPO_ROOT)
    print("Models to run:", ", ".join(requested_models))

    for model_name in requested_models:
        model = MODEL_COMMANDS[model_name]
        command = command_for_model(model, args.skip_heavy)
        print("\n" + "=" * 80)
        print(f"Running {model.name}: {' '.join(command)}")
        print("=" * 80)
        start_time = time.time()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)  # noqa: S603 - known local scripts
        elapsed = time.time() - start_time
        status = "succeeded" if completed.returncode == 0 else "failed"
        results.append(
            {
                "model": model.name,
                "status": status,
                "return_code": completed.returncode,
                "elapsed_seconds": elapsed,
            }
        )
        print(f"{model.name} {status} in {elapsed:.2f}s with return code {completed.returncode}.")

    print("\n" + "=" * 80)
    print("Run summary")
    print("=" * 80)
    for result in results:
        print(
            f"- {result['model']}: {result['status']} "
            f"(return_code={result['return_code']}, elapsed={result['elapsed_seconds']:.2f}s)"
        )

    failed = [result for result in results if result["return_code"] != 0]
    if failed:
        print("\nOne or more models failed. Check messages above; missing local data is a common cause.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
