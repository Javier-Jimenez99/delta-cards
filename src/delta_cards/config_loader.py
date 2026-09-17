"""Configuration loader for delta-cards."""

import json
from pathlib import Path
from typing import List, Union
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
        project_root = Path(__file__).resolve().parent.parent.parent
        alt_path = (project_root / config_path).resolve()
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration file '{path}': {exc}") from exc

    if "game" not in data:
        name_lower = str(data.get("name", "")).lower()
        filename_lower = path.name.lower()
        if "magic" in name_lower or "magic" in filename_lower or "mtg" in name_lower:
            data["game"] = "magic"
        else:
            data["game"] = "pokemon"

    return SearchConfig.model_validate(data)


def load_all_configs(config_dir: Union[str, Path] = "config") -> List[SearchConfig]:
    """Loads all configuration JSON files directly located in config_dir.

    Subdirectories (such as extra_config) are intentionally not searched.

    Args:
        config_dir: Directory path containing config JSON files.

    Returns:
        List[SearchConfig]: List of validated SearchConfig instances.
    """
    path = Path(config_dir).expanduser().resolve()
    if not path.is_dir():
        project_root = Path(__file__).resolve().parent.parent.parent
        alt_path = (project_root / config_dir).resolve()
        if alt_path.is_dir():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration directory not found: {path}")

    configs: List[SearchConfig] = []
    # Only load direct .json files in this folder, not recursive subdirectories
    for item in sorted(path.glob("*.json")):
        if item.is_file():
            configs.append(load_config(item))
    return configs
