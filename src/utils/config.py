"""Configuration loading utilities.

The project prefers PyYAML when it is installed. A small fallback parser is
included so Phase 1 smoke tests can run in a bare Python environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL_SECTIONS = {
    "project",
    "simulation",
    "optimization",
    "ml",
    "experiments",
    "paths",
    "logging",
}


def get_project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the project YAML configuration file."""

    config_path = Path(path) if path else get_project_root() / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    config = _load_yaml_text(text)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required Phase 1 configuration sections."""

    missing = sorted(REQUIRED_TOP_LEVEL_SECTIONS.difference(config))
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    duration = config["simulation"].get("duration_s", 0)
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("simulation.duration_s must be a positive number")

    train_ratio = config["ml"].get("train_ratio", 0)
    test_ratio = config["ml"].get("test_ratio", 0)
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("ml.train_ratio and ml.test_ratio must be positive")


def _load_yaml_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _simple_yaml_load(text)

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration must be a mapping")
    return loaded


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the limited YAML subset used by config/config.yaml."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if ":" not in raw_line:
            raise ValueError(f"Unsupported config line: {raw_line!r}")

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = raw_line.strip().split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation near key: {key}")

        parent = stack[-1][1]
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)

    return root


def _coerce_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_coerce_scalar(item.strip()) for item in body.split(",")]

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value

