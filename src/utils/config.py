from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    config_path = Path(path) if path else project_root() / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value, root=None) -> Path:
    root = Path(root) if root is not None else project_root()
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def config_path(config: dict, *keys, root=None) -> Path:
    value = config
    for key in keys:
        value = value[key]
    return resolve_path(value, root=root)
