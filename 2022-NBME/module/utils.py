import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch


def seed_everything(seed: int = 0):
    """シード値を固定する

    Args:
        seed (int, optional): 乱数固定に使用するシード値. Defaults to 0.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(
    module: str, filepath: Union[str, Path] = "train.log", level: int = logging.INFO
):
    """logger取得用関数

    Args:
        module (str): loggerを呼び出すモジュール名
        filepath (Union[str, Path], optional): ログを保存するパス. Defaults to "train.log".
        level (int, optional): ログのレベル. Defaults to logging.INFO (20).

    Returns:
        logger: logger
    """
    logger = logging.getLogger(module)
    logger.setLevel(level)
    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(message)s"))
    handler2 = logging.FileHandler(filename=filepath)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class AverageMeter(object):
    """Lossなどの平均を監視する"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: float) -> str:
    """秒数を分と秒に直す

    Args:
        s (float): 秒数

    Returns:
        str: `{%d}m {%d}s`表記の時間
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since: float, percent: float) -> str:
    """経過時間と予測残時間を出力する

    Args:
        since (float): 経過時間
        percent (int): 経過した割合

    Returns:
        str: `{%s} (remain {%s})`表記の経過時間と予測残時間
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
