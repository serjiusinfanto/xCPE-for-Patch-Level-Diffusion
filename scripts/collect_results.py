"""
collect_results.py — Aggregate fine-tuning logs into a results CSV.

Reads finetune CSV logs written by src/utils/logging.py under results/logs/.
Each finetune CSV ends with a test row: step=0, test_mse, test_mae.
Aggregates over seeds (if present) and writes results/tables/<dataset>_results.csv.

Actual log filename format (from src/utils/logging.py):
    {DATASET}_h{HORIZON}_{VARIANT}_finetune.csv
    {DATASET}_h{HORIZON}_{VARIANT}_seed{SEED}_finetune.csv  (multi-seed runs)

Usage:
    python scripts/collect_results.py --dataset ETTh1
    python scripts/collect_results.py             # all datasets found
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

LOGS_DIR   = Path("results/logs")
TABLES_DIR = Path("results/tables")

# Matches both single-seed and multi-seed filenames
LOG_PATTERN = re.compile(
    r"(?P<dataset>[^_]+)_h(?P<horizon>\d+)_(?P<variant>.+?)(?:_seed(?P<seed>\d+))?_finetune\.csv$"
)


def read_test_row(log_path: Path) -> dict | None:
    """Return test_mse and test_mae from the final row (step == 0)."""
    try:
        with open(log_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        if not rows:
            return None
        last = rows[-1]
        # Test row written as: 0, test_mse, test_mae  (step=0 sentinel)
        if last[0] == "0" and len(last) >= 3:
            return {"test_mse": float(last[1]), "test_mae": float(last[2])}
        # Fallback: look for val_mse in the last training row as a proxy
        if "val_mse" in header and len(last) > header.index("val_mse"):
            idx = header.index("val_mse")
            return {"test_mse": float(last[idx]), "test_mae": float(last[header.index("val_mae")])}
    except Exception as e:
        print(f"  Warning: could not read {log_path.name}: {e}")
    return None


def collect(dataset: str) -> list[dict]:
    records = []
    for log_path in sorted(LOGS_DIR.glob("*.csv")):
        m = LOG_PATTERN.match(log_path.name)
        if m is None:
            continue
        if m.group("dataset").lower() != dataset.lower():
            continue
        row = read_test_row(log_path)
        if row is None:
            print(f"  Skipping {log_path.name}: no test row found")
            continue
        records.append({
            "variant":  m.group("variant"),
            "dataset":  m.group("dataset"),
            "horizon":  int(m.group("horizon")),
            "seed":     int(m.group("seed") or 42),
            "test_mse": row["test_mse"],
            "test_mae": row["test_mae"],
        })
    return records


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for r in records:
        groups.setdefault((r["variant"], r["horizon"]), []).append(r)

    rows = []
    for (variant, horizon), group in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        mses = [g["test_mse"] for g in group]
        maes = [g["test_mae"] for g in group]
        n = len(mses)
        rows.append({
            "variant":  variant,
            "horizon":  horizon,
            "n_seeds":  n,
            "mse_mean": round(statistics.mean(mses), 4),
            "mse_std":  round(statistics.stdev(mses), 4) if n > 1 else 0.0,
            "mae_mean": round(statistics.mean(maes), 4),
            "mae_std":  round(statistics.stdev(maes), 4) if n > 1 else 0.0,
        })
    return rows


def write_table(rows: list[dict], dataset: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / f"{dataset.lower()}_results.csv"
    fieldnames = ["variant", "horizon", "n_seeds",
                  "mse_mean", "mse_std", "mae_mean", "mae_std"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_table(rows: list[dict], dataset: str) -> None:
    print(f"\n{'='*68}")
    print(f"  Results — {dataset}")
    print(f"{'='*68}")
    print(f"{'Variant':<16} {'H':>4}  {'MSE':>8} {'±':>6}  {'MAE':>8} {'±':>6}  N")
    print("-" * 68)
    cur_variant = None
    for r in rows:
        if r["variant"] != cur_variant:
            if cur_variant is not None:
                print()
            cur_variant = r["variant"]
        print(
            f"{r['variant']:<16} {r['horizon']:>4}  "
            f"{r['mse_mean']:>8.4f} {r['mse_std']:>6.4f}  "
            f"{r['mae_mean']:>8.4f} {r['mae_std']:>6.4f}  {r['n_seeds']}"
        )
    print(f"{'='*68}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted({
            m.group("dataset")
            for f in LOGS_DIR.glob("*.csv")
            if (m := LOG_PATTERN.match(f.name)) is not None
        })
        if not datasets:
            print(f"No matching log files found in {LOGS_DIR}/")
            return

    for dataset in datasets:
        print(f"\nCollecting results for {dataset} …")
        records = collect(dataset)
        if not records:
            print(f"  No completed runs found for {dataset}.")
            continue
        rows = aggregate(records)
        out_path = write_table(rows, dataset)
        print_table(rows, dataset)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
