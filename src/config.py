"""Shared configuration loading for the pipeline scripts."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config():
    """Load and parse config.yaml from the repository root."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
