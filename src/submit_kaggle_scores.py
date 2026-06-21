"""Submit completed stage predictions to Kaggle and store official scores."""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


COMPETITION = "home-credit-default-risk"
KAGGLE_COLUMNS = [
    "kaggle_public_auc",
    "kaggle_private_auc",
    "kaggle_submission_date",
    "kaggle_submission_status",
    "kaggle_submission_description",
    "kaggle_file_name",
]


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    summary_path = results_dir / "summary.csv"
    summary = _read_summary(summary_path)

    kaggle_bin = None
    if not args.dry_run:
        kaggle_bin = _resolve_kaggle_bin(args.kaggle_bin)
        if not _has_kaggle_credentials():
            raise SystemExit(
                "Kaggle credentials were not found. Put kaggle.json in "
                f"{Path.home() / '.kaggle' / 'kaggle.json'}, keep OAuth credentials.json, "
                "or set KAGGLE_USERNAME and KAGGLE_KEY."
            )
        if _refresh_existing_submissions(summary, kaggle_bin, args.competition):
            _write_summary(summary_path, summary)

    candidates = _submission_candidates(summary)
    if args.limit is not None:
        candidates = candidates.head(args.limit)
    if candidates.empty:
        print("No completed submissions need Kaggle scoring.")
        return

    print("Submission order:")
    for _, row in candidates.iterrows():
        print(
            f"- {row['stage']}/{row['model']} "
            f"oof={float(row['oof_auc']):.6f} file={_submission_path(row)}"
        )

    if args.dry_run:
        return

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    completed_indices: list[int] = []

    for index, row in candidates.iterrows():
        message = _submission_message(row, run_id)
        path = _submission_path(row)
        print(f"Submitting {row['stage']}/{row['model']}: {path}")
        submit_result = _run_command(
            [
                str(kaggle_bin),
                "competitions",
                "submit",
                args.competition,
                "-f",
                str(path),
                "-m",
                message,
            ]
        )

        if submit_result.returncode != 0:
            print(submit_result.stdout)
            print(submit_result.stderr, file=sys.stderr)
            _print_remaining(candidates, completed_indices)
            break

        record = _poll_for_submission(
            kaggle_bin=kaggle_bin,
            competition=args.competition,
            message=message,
            timeout_seconds=args.poll_timeout,
            interval_seconds=args.poll_interval,
        )
        if record is None:
            record = {
                "description": message,
                "status": "poll_timeout",
                "date": "",
                "fileName": path.name,
                "publicScore": "",
                "privateScore": "",
            }
        _update_summary_row(summary, index, record)
        _write_summary(summary_path, summary)
        completed_indices.append(index)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit completed stage predictions to Kaggle.")
    parser.add_argument("--results-dir", default="results", help="Directory containing summary.csv and stage outputs")
    parser.add_argument("--competition", default=COMPETITION, help="Kaggle competition slug")
    parser.add_argument("--kaggle-bin", default=None, help="Path to kaggle executable")
    parser.add_argument("--poll-timeout", type=int, default=600, help="Seconds to wait for Kaggle scoring")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between submission status checks")
    parser.add_argument("--limit", type=int, default=None, help="Submit at most this many rows")
    parser.add_argument("--dry-run", action="store_true", help="List planned submissions without calling Kaggle")
    return parser.parse_args()


def _read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing summary file: {path}")
    summary = pd.read_csv(path)
    for column in KAGGLE_COLUMNS:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary


def _submission_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    mask = summary.apply(lambda row: _submission_path(row).exists(), axis=1)
    score_missing = summary["kaggle_public_auc"].isna() & summary["kaggle_private_auc"].isna()
    not_submitted = summary["kaggle_submission_description"].isna() | summary["kaggle_submission_description"].eq("")
    candidates = summary[mask & score_missing & not_submitted].copy()
    return candidates.sort_values(["oof_auc", "stage", "model"], ascending=[False, True, True])


def _submission_path(row: pd.Series) -> Path:
    stage_submission_files = {
        "s6_avg": "submission_simple_average.csv",
        "s6_stack": "submission_logistic_stack.csv",
    }
    file_name = stage_submission_files.get(str(row["stage"]), "submission.csv")
    return Path(str(row["output_dir"])) / file_name


def _submission_message(row: pd.Series, run_id: str) -> str:
    return (
        f"hcr {row['stage']}/{row['model']} "
        f"oof={float(row['oof_auc']):.6f} features={int(row['n_features'])} run={run_id}"
    )


def _has_kaggle_credentials() -> bool:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    oauth_path = Path.home() / ".kaggle" / "credentials.json"
    env_credentials = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    return token_path.exists() or oauth_path.exists() or bool(env_credentials)


def _resolve_kaggle_bin(value: str | None) -> Path:
    if value is not None:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"kaggle executable not found: {path}")
        return path

    env_value = os.environ.get("KAGGLE_BIN")
    if env_value:
        path = Path(env_value)
        if not path.exists():
            raise FileNotFoundError(f"KAGGLE_BIN points to a missing file: {path}")
        return path

    local = Path(".venv") / "Scripts" / "kaggle.exe"
    if local.exists():
        return local
    return Path("kaggle")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _poll_for_submission(
    kaggle_bin: Path,
    competition: str,
    message: str,
    timeout_seconds: int,
    interval_seconds: int,
) -> dict[str, str] | None:
    deadline = time.monotonic() + timeout_seconds
    last_record: dict[str, str] | None = None

    while True:
        result = _run_command([str(kaggle_bin), "competitions", "submissions", "-v", competition])
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return last_record

        records = _parse_submissions_csv(result.stdout)
        for record in records:
            if record.get("description") == message:
                last_record = record
                if record.get("publicScore") or record.get("privateScore"):
                    return record

        if time.monotonic() >= deadline:
            return last_record
        time.sleep(interval_seconds)


def _parse_submissions_csv(output: str) -> list[dict[str, str]]:
    lines = [line for line in output.splitlines() if line.strip()]
    header_index = next(
        (i for i, line in enumerate(lines) if line.startswith("fileName,") or line.startswith("ref,")),
        None,
    )
    if header_index is None:
        return []
    csv_text = "\n".join(lines[header_index:])
    return list(csv.DictReader(io.StringIO(csv_text)))


def _refresh_existing_submissions(summary: pd.DataFrame, kaggle_bin: Path, competition: str) -> bool:
    pending = summary[
        summary["kaggle_submission_description"].notna()
        & summary["kaggle_submission_description"].ne("")
        & summary["kaggle_public_auc"].isna()
        & summary["kaggle_private_auc"].isna()
    ]
    if pending.empty:
        return False

    result = _run_command([str(kaggle_bin), "competitions", "submissions", "-v", competition])
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return False

    records = {
        record.get("description"): record
        for record in _parse_submissions_csv(result.stdout)
        if record.get("description")
    }
    updated = False
    for index, row in pending.iterrows():
        record = records.get(row["kaggle_submission_description"])
        if record is None:
            continue
        _update_summary_row(summary, index, record)
        updated = True
        print(f"Refreshed Kaggle score for {row['stage']}/{row['model']}.")
    return updated


def _update_summary_row(summary: pd.DataFrame, index: int, record: dict[str, str]) -> None:
    summary.loc[index, "kaggle_public_auc"] = _score_or_na(record.get("publicScore", ""))
    summary.loc[index, "kaggle_private_auc"] = _score_or_na(record.get("privateScore", ""))
    summary.loc[index, "kaggle_submission_date"] = record.get("date", "")
    summary.loc[index, "kaggle_submission_status"] = record.get("status", "")
    summary.loc[index, "kaggle_submission_description"] = record.get("description", "")
    summary.loc[index, "kaggle_file_name"] = record.get("fileName", "")


def _score_or_na(value: str) -> object:
    value = str(value).strip()
    if not value or value.lower() == "none":
        return pd.NA
    return float(value)


def _write_summary(path: Path, summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


def _print_remaining(candidates: pd.DataFrame, completed_indices: list[int]) -> None:
    remaining = candidates.drop(index=completed_indices, errors="ignore")
    if remaining.empty:
        return
    print("Remaining unsubmitted stages:")
    for _, row in remaining.iterrows():
        print(f"- {row['stage']}/{row['model']}")


if __name__ == "__main__":
    main()
