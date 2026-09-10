"""Configuration loader for delta-cards."""

import json
from pathlib import Path
from typing import Union
from .models import SearchConfig


def load_config(config_path: Union[str, Path]) -> SearchConfig:
    """Loads and validates a search configuration JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        SearchConfig: Validated search configuration object.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If JSON syntax or model validation fails.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration file '{path}': {exc}") from exc

    return SearchConfig.model_validate(data)
