import yaml


def load_config(file_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Params
    file_path: path to a YAML file

    Returns
    conf: dictionary with config values
    """
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf
