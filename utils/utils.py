import yaml
import torch

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



def save_ckp(checkpoint, flag, checkpoint_path, best_model_path, epoch):
    torch.save(
        checkpoint,
        f"{checkpoint_path}{epoch}.pt"
    )
    if flag:
        torch.save(
        checkpoint,
        best_model_path
    )