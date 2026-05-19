import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import load_edgeiiot_dataset, load_prepared_csv
from src.utils.config import config_path, load_config


def save_frame(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    print(f"Saved {path} shape={df.shape}")


def preprocess_ciciot(config, output_dir):
    dataset_cfg = config["datasets"]["ciciot"]
    csv_path = config_path(config, "datasets", "ciciot", "merged_csv", root=ROOT)
    sample_rows = dataset_cfg.get("sample_rows")
    df, label_col = load_prepared_csv(
        csv_path,
        sample_rows=sample_rows if sample_rows and sample_rows > 0 else None,
        seed=config.get("seed", 42),
        dataset_name="CICIoT2023-main",
    )
    save_frame(df, output_dir / "ciciot2023_prepared.parquet")
    return label_col


def preprocess_edgeiiot(config, output_dir):
    dataset_cfg = config["datasets"]["edgeiiot"]
    csv_path = config_path(config, "datasets", "edgeiiot", "csv_path", root=ROOT)
    sample_rows = dataset_cfg.get("sample_rows")
    df, label_col = load_edgeiiot_dataset(
        csv_path,
        sample_rows=sample_rows if sample_rows and sample_rows > 0 else None,
        seed=config.get("seed", 42),
        drop_duplicates=bool(dataset_cfg.get("drop_duplicates", True)),
        use_author_style_drops=bool(dataset_cfg.get("use_author_style_drops", True)),
    )
    save_frame(df, output_dir / "edgeiiot_prepared.parquet")
    return label_col


def main():
    parser = argparse.ArgumentParser(description="Prepare CICIoT2023 and Edge-IIoTset files for local runs.")
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--dataset", choices=["ciciot", "edgeiiot", "both"], default="ciciot")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config_path(config, "paths", "processed_dir", root=ROOT)

    if args.dataset in {"ciciot", "both"}:
        preprocess_ciciot(config, output_dir)
    if args.dataset in {"edgeiiot", "both"}:
        preprocess_edgeiiot(config, output_dir)


if __name__ == "__main__":
    main()
