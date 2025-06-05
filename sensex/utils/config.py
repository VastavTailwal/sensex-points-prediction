import yaml
from typing import Any


def load_config(file_path: str) -> dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Parameters
    ----------
    file_path : str
        Path to a YAML file.

    Returns
    -------
    conf : dict
        Dictionary with config values.
    """
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf
