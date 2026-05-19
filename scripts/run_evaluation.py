import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import load_edgeiiot_dataset, load_prepared_csv, split_70_15_15, transform_external_features
from src.evaluation.metrics import compute_metrics
from src.models.models import TinyMLP
from src.training.train import predict_model
from src.utils.config import config_path, load_config


def load_dataset(config, dataset):
    if dataset == "ciciot":
        cfg = config["datasets"]["ciciot"]
        csv_path = config_path(config, "datasets", "ciciot", "merged_csv", root=ROOT)
        return load_prepared_csv(
            csv_path,
            sample_rows=cfg.get("sample_rows"),
            seed=config.get("seed", 42),
            dataset_name="CICIoT2023-main",
        )

    cfg = config["datasets"]["edgeiiot"]
    csv_path = config_path(config, "datasets", "edgeiiot", "csv_path", root=ROOT)
    return load_edgeiiot_dataset(
        csv_path,
        sample_rows=cfg.get("sample_rows"),
        seed=config.get("seed", 42),
        drop_duplicates=bool(cfg.get("drop_duplicates", True)),
        use_author_style_drops=bool(cfg.get("use_author_style_drops", True)),
    )


def load_model(bundle):
    model_path = Path(bundle["model_path"])
    if bundle["model_name"] == "tinymlp":
        model = TinyMLP(bundle["input_dim"], bundle["output_dim"])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    return joblib.load(model_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model produced by scripts/run_training.py.")
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--dataset", choices=["ciciot", "edgeiiot"], default="ciciot")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--model", choices=["lr", "rf", "lgbm", "tinymlp"], default="lgbm")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    stem = f"{args.dataset}_{args.task}_{args.model}_seed{seed}"
    bundle_path = config_path(config, "paths", "models_dir", root=ROOT) / "local" / f"{stem}_preprocess.joblib"
    bundle = joblib.load(bundle_path)

    df, label_col = load_dataset(config, args.dataset)
    if args.task == "multiclass":
        df = df[df["Category"] != "Benign"].copy().reset_index(drop=True)
        y_split = df["Category"].astype(str).to_numpy()
    else:
        y_split = df["BinaryLabel"].to_numpy()

    _, _, test_idx = split_70_15_15(y_split, seed)
    X_test, y_test = transform_external_features(
        df.iloc[test_idx],
        label_col,
        bundle["feature_cols"],
        bundle["scaler"],
        bundle["medians"],
    )
    if args.task == "multiclass":
        y_test = bundle["encoder"].transform(df.iloc[test_idx]["Category"].astype(str).to_numpy())

    model = load_model(bundle)
    pred, probs = predict_model(args.model, model, X_test)
    metrics = compute_metrics(y_test, pred, probs, args.task)

    output_path = config_path(config, "paths", "results_dir", root=ROOT) / f"{stem}_evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    print(f"Saved evaluation: {output_path}")


if __name__ == "__main__":
    main()
