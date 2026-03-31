"""
Compute dataset statistics for FalconCode train/test splits and save a LaTeX table.

Usage:
    python scripts/dataset_stats.py \
        --train <path/to/train/dataframe.csv> \
        --test  <path/to/test/dataframe.csv>  \
        [--output outputs/tables/dataset_stats_table.tex]
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def get_unit_test_score(testcase_output: str) -> float:
    lines = testcase_output.splitlines()
    utr = [l for l in lines if l.startswith("Unit Test Returned:")]
    if utr:
        return float(utr[0].replace("Unit Test Returned:", "").strip())
    return 0.0


def extract_final_grade(messages_str) -> float | None:
    try:
        messages = ast.literal_eval(messages_str) if isinstance(messages_str, str) else messages_str
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return get_unit_test_score(msg.get("content", ""))
        return None
    except Exception as e:
        print(f"Warning: could not parse message — {e}")
        return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(df: pd.DataFrame, split_name: str = "Train") -> dict:
    stats = {
        "n_traj": len(df),
        "n_students": df["user_id"].nunique(),
        "n_assignments": df["problem_id"].nunique(),
        "avg_len": df["traj_len"].mean(),
    }

    print(f"  Extracting grades from {split_name}...")
    df = df.copy()
    df["final_grade"] = df["messages"].apply(extract_final_grade)
    df["success"] = df["final_grade"].apply(lambda x: x is not None and x >= 0.999)

    stats["n_success"] = int(df["success"].sum())
    stats["n_fail"] = len(df) - stats["n_success"]

    valid = df["final_grade"].dropna()
    stats["avg_grade"] = float(valid.mean()) if len(valid) else 0.0
    stats["med_grade"] = float(valid.median()) if len(valid) else 0.0

    return stats


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def generate_latex(train_stats: dict, test_stats: dict) -> str:
    def fmt(n):
        return str(int(n))

    header = r"""\begin{table}[h]
\centering
\caption{\textbf{FalconCode dataset statistics.} Summary of training and test splits.
  \#Traj: total trajectories; \#Stud: unique students;
  \#Succ/\#Fail: trajectories with final grade 100\%/below 100\%;
  \#Asg: unique programming assignments;
  Avg.\,Len: mean submissions per trajectory;
  Avg.\,G and Med.\,G: mean and median final grades.}
\label{tab:dataset_stats}
\resizebox{\columnwidth}{!}{%
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{lcccccccc}
\toprule
Split & \#Traj & \#Stud & \#Succ & \#Fail & \#Asg & Avg.\,Len & Avg.\,G & Med.\,G \\
\midrule
"""

    rows = ""
    for name, s in [("Train", train_stats), ("Test", test_stats)]:
        rows += (
            f"{name} & {fmt(s['n_traj'])} & {fmt(s['n_students'])} & "
            f"{fmt(s['n_success'])} & {fmt(s['n_fail'])} & "
            f"{s['n_assignments']} & {s['avg_len']:.1f} & "
            f"{s['avg_grade']:.2f} & {s['med_grade']:.2f} \\\\\n"
        )

    footer = r"""\bottomrule
\end{tabular}
}
\end{table}"""

    return header + rows + footer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate FalconCode dataset statistics table.")
    parser.add_argument("--train", required=True, help="Path to train dataframe.csv")
    parser.add_argument("--test", required=True, help="Path to test dataframe.csv")
    parser.add_argument(
        "--output",
        default="outputs/tables/dataset_stats_table.tex",
        help="Output path for the LaTeX table (default: outputs/tables/dataset_stats_table.tex)",
    )
    args = parser.parse_args()

    print("Loading datasets...")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    print("\nComputing statistics...")
    train_stats = compute_statistics(train_df, "Train")
    test_stats = compute_statistics(test_df, "Test")

    for name, s in [("Train", train_stats), ("Test", test_stats)]:
        print(f"\n{name}:")
        for k, v in s.items():
            print(f"  {k:20s}: {v:.2f}" if isinstance(v, float) else f"  {k:20s}: {v:,}")

    latex = generate_latex(train_stats, test_stats)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(latex)
    print(f"\nSaved LaTeX table to: {args.output}")


if __name__ == "__main__":
    main()
