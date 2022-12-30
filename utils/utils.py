import yaml


def load_yaml(path: str) -> dict:
    """yamlファイルを読み込んで辞書を返す．

    Args:
        path (str): yamlファイルのパス

    Returns:
        Dict: yamlファイルの中身
    """
    with open(path, "r") as yml:
        config = yaml.safe_load(yml)

    return config