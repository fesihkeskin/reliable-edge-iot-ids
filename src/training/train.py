from dataclasses import dataclass
import gc
from pathlib import Path
import random
import time

import joblib
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.models import TinyMLP


@dataclass
class TrainingSettings:
    fast_mode: bool = False
    tinymlp_epochs: int = 25
    tinymlp_fast_epochs: int = 12
    tinymlp_batch_size: int = 8192
    rf_n_estimators: int = 250
    rf_fast_n_estimators: int = 120
    rf_max_depth: int = 22
    rf_fast_max_depth: int = 16
    lgbm_n_estimators: int = 300
    lgbm_fast_n_estimators: int = 160
    run_lr: bool = True
    run_rf: bool = True
    run_lgbm: bool = True
    run_tinymlp: bool = True

    @property
    def effective_tinymlp_epochs(self):
        return self.tinymlp_fast_epochs if self.fast_mode else self.tinymlp_epochs

    @property
    def effective_rf_n_estimators(self):
        return self.rf_fast_n_estimators if self.fast_mode else self.rf_n_estimators

    @property
    def effective_rf_max_depth(self):
        return self.rf_fast_max_depth if self.fast_mode else self.rf_max_depth

    @property
    def effective_lgbm_n_estimators(self):
        return self.lgbm_fast_n_estimators if self.fast_mode else self.lgbm_n_estimators


def settings_from_config(config):
    training = config.get("training", {})
    models = config.get("models", {})
    return TrainingSettings(
        fast_mode=bool(training.get("fast_mode", False)),
        tinymlp_epochs=int(training.get("tinymlp_epochs", 25)),
        tinymlp_fast_epochs=int(training.get("tinymlp_fast_epochs", 12)),
        tinymlp_batch_size=int(training.get("tinymlp_batch_size", 8192)),
        rf_n_estimators=int(models.get("rf_n_estimators", 250)),
        rf_fast_n_estimators=int(models.get("rf_fast_n_estimators", 120)),
        rf_max_depth=int(models.get("rf_max_depth", 22)),
        rf_fast_max_depth=int(models.get("rf_fast_max_depth", 16)),
        lgbm_n_estimators=int(models.get("lgbm_n_estimators", 300)),
        lgbm_fast_n_estimators=int(models.get("lgbm_fast_n_estimators", 160)),
        run_lr=bool(models.get("run_lr", True)),
        run_rf=bool(models.get("run_rf", True)),
        run_lgbm=bool(models.get("run_lgbm", True)),
        run_tinymlp=bool(models.get("run_tinymlp", True)),
    )


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def memory_report():
    print(f"CPU RSS: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


def train_tiny_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    seed,
    epochs=None,
    settings=None,
    device=None,
):
    settings = settings or TrainingSettings()
    device = device or get_device()
    set_global_seed(seed)
    epochs = epochs or settings.effective_tinymlp_epochs
    model = TinyMLP(X_train.shape[1], num_classes).to(device)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(
        dataset,
        batch_size=settings.tinymlp_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_state, best_f1, bad = None, -1.0, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(X_val_t), dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if settings.fast_mode and bad >= 4:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_torch(model, X, batch_size=65536):
    model.eval()
    device = next(model.parameters()).device
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.vstack(probs)
    return probs.argmax(axis=1), probs


def predict_torch_cpu(model, X, batch_size=65536):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.vstack(probs)
    return probs.argmax(axis=1), probs


def fit_classic_model(model_name, X_train, y_train, seed, settings=None):
    settings = settings or TrainingSettings()
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training data has fewer than two classes.")
    if model_name == "lr":
        model = LogisticRegression(max_iter=500, solver="saga", n_jobs=-1, random_state=seed)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=settings.effective_rf_n_estimators,
            max_depth=settings.effective_rf_max_depth,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    elif model_name == "lgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=settings.effective_lgbm_n_estimators,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(model_name)
    model.fit(X_train, y_train)
    return model


def predict_model(model_name, model, X):
    if model_name == "tinymlp":
        return predict_torch(model, X)
    pred = model.predict(X)
    probs = model.predict_proba(X)
    return pred, probs


def save_model_size(model_name, model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "tinymlp" or isinstance(model, nn.Module):
        torch.save(model.state_dict(), path)
    else:
        joblib.dump(model, path)
    return path.stat().st_size / (1024**2)


def save_torch_size(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path.stat().st_size / 1024**2


def benchmark_inference(model_name, model, X, repeat=3):
    predict_model(model_name, model, X[: min(len(X), 2048)])
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        predict_model(model_name, model, X)
        times.append(time.perf_counter() - start)
    elapsed = min(times)
    return {
        "latency_per_flow_sec": elapsed / max(len(X), 1),
        "throughput_flows_sec": len(X) / max(elapsed, 1e-9),
        "peak_rss_mb": psutil.Process().memory_info().rss / 1024**2,
    }


def benchmark_torch_variant(model, X, quantized=False, repeat=5):
    pred_fn = predict_torch_cpu if quantized else predict_torch
    pred_fn(model, X[: min(2048, len(X))])
    times = []
    pred, probs = None, None
    for _ in range(repeat):
        start = time.perf_counter()
        pred, probs = pred_fn(model, X)
        times.append(time.perf_counter() - start)
    elapsed = min(times)
    return pred, probs, {
        "latency_per_flow_sec": elapsed / max(len(X), 1),
        "throughput_flows_sec": len(X) / max(elapsed, 1e-9),
        "peak_rss_mb": psutil.Process().memory_info().rss / 1024**2,
    }


def model_names_to_run(settings=None):
    settings = settings or TrainingSettings()
    out = []
    if settings.run_lr:
        out.append("lr")
    if settings.run_rf:
        out.append("rf")
    if settings.run_lgbm:
        out.append("lgbm")
    if settings.run_tinymlp:
        out.append("tinymlp")
    return out
