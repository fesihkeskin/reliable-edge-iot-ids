import argparse
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import (
    fit_transform_features,
    load_edgeiiot_dataset,
    load_prepared_csv,
    split_70_15_15,
)
from src.evaluation.metrics import compute_metrics
from src.training.train import (
    fit_classic_model,
    predict_model,
    save_model_size,
    settings_from_config,
    train_tiny_mlp,
)
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


def main():
    parser = argparse.ArgumentParser(description="Train one local baseline model.")
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--dataset", choices=["ciciot", "edgeiiot"], default="ciciot")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--model", choices=["lr", "rf", "lgbm", "tinymlp"], default="lgbm")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = settings_from_config(config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    df, label_col = load_dataset(config, args.dataset)

    if args.task == "multiclass":
        df = df[df["Category"] != "Benign"].copy().reset_index(drop=True)
        y_split = df["Category"].astype(str).to_numpy()
    else:
        y_split = df["BinaryLabel"].to_numpy()

    train_idx, val_idx, test_idx = split_70_15_15(y_split, seed)
    X_train, y_train, X_val, y_val, X_test, y_test, encoder, scaler, medians, feature_cols = fit_transform_features(
        df,
        label_col,
        args.task,
        train_idx,
        val_idx,
        test_idx,
    )

    start = time.perf_counter()
    if args.model == "tinymlp":
        model = train_tiny_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            len(np.unique(y_train)),
            seed,
            settings=settings,
        )
    else:
        model = fit_classic_model(args.model, X_train, y_train, seed, settings=settings)
    train_sec = time.perf_counter() - start

    pred, probs = predict_model(args.model, model, X_test)
    metrics = compute_metrics(y_test, pred, probs, args.task)

    model_dir = config_path(config, "paths", "models_dir", root=ROOT) / "local"
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.task}_{args.model}_seed{seed}"
    model_path = model_dir / (stem + (".pt" if args.model == "tinymlp" else ".joblib"))
    metrics["model_size_mb"] = save_model_size(args.model, model, model_path)
    metrics["train_sec"] = train_sec

    bundle = {
        "dataset": args.dataset,
        "task": args.task,
        "model_name": args.model,
        "seed": seed,
        "label_col": label_col,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "medians": medians,
        "encoder": encoder,
        "model_path": str(model_path),
        "input_dim": X_train.shape[1],
        "output_dim": len(np.unique(y_train)),
    }
    joblib.dump(bundle, model_dir / f"{stem}_preprocess.joblib")

    metrics_path = config_path(config, "paths", "results_dir", root=ROOT) / f"{stem}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
