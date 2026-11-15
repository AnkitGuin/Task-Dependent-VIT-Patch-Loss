"""Machine-readable source rows serve as the canonical result tables."""

from pathlib import Path
import pandas as pd


def create_tables(input_dir="reports", output_dir="reports/tables"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(input_dir) / "experiments.csv")
    wrote = []
    for eid, filename in [
        ("E02", "T3_gaussian"),
        ("E06", "T4_embeddings"),
        ("E09", "T4_models"),
        ("E07", "T5_antialiasing"),
        ("E08", "T6_overlap"),
    ]:
        sub = df[df.experiment_id == eid]
        if not sub.empty:
            sub.to_csv(output / (filename + ".csv"), index=False)
            wrote.append(filename)
    return wrote
