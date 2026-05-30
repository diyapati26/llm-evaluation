"""YAML config loader."""
import yaml


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def flatten_models(cfg_models):
    """{provider: [model, ...]}  ->  ['provider:model', ...]"""
    return [f"{p}:{m}" for p, ms in (cfg_models or {}).items() for m in (ms or [])]


def models_from_config(cfg):
    return flatten_models(cfg.get("models"))
