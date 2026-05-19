from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def result_exists(path, skip_if_exists=True):
    path = Path(path)
    return bool(skip_if_exists and path.exists() and path.stat().st_size > 0)


def read_csv_maybe_sample(path, sample_rows=None, seed=42, chunksize=250_000):
    path = str(path)
    if sample_rows is None or sample_rows <= 0:
        return pd.read_csv(path, low_memory=False)

    rng = np.random.default_rng(seed)
    selected = None
    for chunk in tqdm(
        pd.read_csv(path, chunksize=chunksize, low_memory=False),
        desc=f"Sampling {Path(path).name}",
    ):
        chunk = chunk.copy()
        chunk["__sample_key"] = rng.random(len(chunk), dtype=np.float32)
        if selected is None:
            selected = chunk.nsmallest(sample_rows, "__sample_key")
        else:
            selected = pd.concat([selected, chunk], ignore_index=True)
            if len(selected) > sample_rows + chunksize:
                selected = selected.nsmallest(sample_rows, "__sample_key")

    if selected is None or selected.empty:
        raise ValueError(f"No rows loaded from {path}")
    return (
        selected.nsmallest(sample_rows, "__sample_key")
        .drop(columns=["__sample_key"])
        .reset_index(drop=True)
    )


def write_results(rows, path, extra_cols=None, metric_cols=None, resource_cols=None):
    if metric_cols is None or resource_cols is None:
        from src.evaluation.metrics import METRIC_COLS, RESOURCE_COLS

        metric_cols = METRIC_COLS if metric_cols is None else metric_cols
        resource_cols = RESOURCE_COLS if resource_cols is None else resource_cols

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    extra_cols = extra_cols or []
    for col in extra_cols + list(metric_cols) + list(resource_cols):
        if col not in df.columns:
            df[col] = np.nan
    df.to_csv(path, index=False)
    print("Saved:", path, "rows=", len(df))
    return df


def iter_csv_files(root):
    for path in Path(root).rglob("*.csv"):
        if path.is_file():
            yield path


def count_csv_files(root):
    root = Path(root)
    return sum(1 for _ in root.rglob("*.csv")) if root.exists() else 0


def detect_ciciot_raw_csv_dir(extract_dir):
    extract_dir = Path(extract_dir)
    candidates = [extract_dir, extract_dir.parent / "MERGED_CSV", extract_dir.parent]
    for candidate in candidates:
        if candidate.exists():
            merged_files = sorted(candidate.glob("Merged*.csv"))
            if merged_files:
                print(
                    f"Detected CICIoT raw CSV folder: {candidate} "
                    f"({len(merged_files)} Merged*.csv files)"
                )
                return candidate
    return extract_dir


def unzip_ciciot_once(zip_path, extract_dir):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    if extract_dir.exists() and count_csv_files(extract_dir) > 0:
        print(f"Already unzipped: {extract_dir}")
        return detect_ciciot_raw_csv_dir(extract_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"CICIoT zip file not found: {zip_path}")
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping once: {zip_path}")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(extract_dir.parent)], check=True)
    return detect_ciciot_raw_csv_dir(extract_dir)


def merge_csvs_streaming(
    input_root,
    output_file,
    max_rows_per_file=None,
    chunksize=500_000,
    skip_if_exists=True,
):
    input_root = Path(input_root)
    output_file = Path(output_file)
    if skip_if_exists and output_file.exists() and output_file.stat().st_size > 0:
        return str(output_file)

    csv_files = sorted(iter_csv_files(input_root))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {input_root}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    for csv_path in tqdm(csv_files, desc="Merging CSV files"):
        rows_written = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            if max_rows_per_file and max_rows_per_file > 0:
                remaining = max_rows_per_file - rows_written
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining]
                rows_written += len(chunk)
            chunk.to_csv(output_file, mode="a", index=False, header=not wrote_header)
            wrote_header = True
    return str(output_file)


def copy_file_to_local_once(source, local, enabled=True, label="file"):
    source_path = Path(source)
    local_path = Path(local)
    if not enabled:
        return str(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"{label} source not found: {source_path}")
    if local_path.exists() and local_path.stat().st_size == source_path.stat().st_size:
        print(f"Local {label} already exists and matches source size.")
        return str(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {label} to local disk: {source_path} -> {local_path}")
    shutil.copy2(source_path, local_path)
    return str(local_path)
