from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import get_logger

logger = get_logger(__name__)

try:
    from transformers.models.deberta_v2.tokenization_deberta_v2_fast import (
        DebertaV2TokenizerFast,
    )
except ModuleNotFoundError:
    logger.info("Failed to import deberta v2 / v3 tokenizer.")
    logger.info(
        "Please set codes and files.\n"
        "See: https://www.kaggle.com/code/librauee/train-deberta-v3-large-baseline?scriptVersionId=88519761&cellId=1"
    )
    raise


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """tokenizerを取得する

    Args:
        model_name (str): モデル名もしくはtokenizerのパス

    Returns:
        PreTrainedTokenizerBase: tokenizer
    """
    if "microsoft/deberta-v" in model_name:
        tokenizer = DebertaV2TokenizerFast.from_pretrained("./deberta-tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trim_offsets=False)
    tokenizer.save_pretrained(Output.tokenizer)
    return tokenizer


def overwrite_model_maxlen(tokenizer: PreTrainedTokenizerBase) -> int:
    """tokenizerの最大長をデータに合わせて上書きする

    Args:
        tokenizer (PreTrainedTokenizerBase): tokenizer

    Returns:
        int: tokenizerの最大長
    """
    patient_notes = pd.read_csv(Input.pt_notes_csv)
    features = pd.read_csv(Input.features_csv)
    max_pn_history_length = max(
        [
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in patient_notes[pt_notes_col.pn_history].fillna("").values
        ]
    )
    max_features_length = max(
        [
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in features[feature_col.feature_text].fillna("").values
        ]
    )
    max_length = max_pn_history_length + max_features_length + 3  # cls + sep + sep
    logger.info(
        f"max_length: {max_pn_history_length} + {max_features_length} + 3 -> {max_length}"
    )
    return max_length


@dataclass
class Model:
    """NLPモデルの設定"""

    name: str = "microsoft/deberta-v3-base"
    dropout_ratio: float = 0.4
    tokenizer: PreTrainedTokenizerBase = get_tokenizer(name)
    max_length: int = overwrite_model_maxlen(tokenizer)


class Schedulers(Enum):
    """使用可能なLRスケジューラ"""

    cosine = 1
    linear = 2


class Optimizers(Enum):
    """使用可能なOptmizer"""

    AdamW = 1


@dataclass
class Optimizer:
    """Optimizerの設定"""

    name: Enum = Optimizers.AdamW
    scheduler: Enum = Schedulers.cosine
    batch_scheduler: bool = True
    num_cycles: float = 0.5
    num_warmup_steps: int = 0
    encoder_lr: float = 2e-5
    decoder_lr: float = 2e-5
    min_lr: float = 1e-6
    eps: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    grad_accum_steps: int = 1
    max_grad_norm: float = 1000.0


@dataclass
class Config:
    """学習時の設定"""

    apex: bool = False
    epochs: int = 1
    batch_size: int = 8
    n_fold: int = 4
    trn_fold: List[int] = [0, 1, 2, 3]
    seed: int = 0
    log_freq: int = 4000
    num_workers: int = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model
    optimizer = Optimizer


@dataclass
class train_col:
    """train.csvの列名"""

    annotation: str = "annotation"
    location: str = "location"
    feature_num: str = "feature_num"
    case_num: str = "case_num"
    pn_num: str = "pn_num"
    # Add
    feature_text: str = "feature_text"
    pn_history: str = "pn_history"
    annotation_length: str = "annotation_length"
    input_lengths: str = "input_lengths"
    batch_max_length: str = "batch_max_length"
    clean_text: str = "clean_text"
    fold: str = "fold"


@dataclass
class feature_col:
    """features.csvの列名"""

    feature_num: str = train_col.feature_num
    case_num: str = train_col.case_num
    feature_text: str = "feature_text"


@dataclass
class pt_notes_col:
    """patient_notes.csvの列名"""

    feature_num: str = train_col.feature_num
    case_num: str = train_col.case_num
    pn_history: str = "pn_history"


@dataclass
class Input:
    """入力ディレクトリ・ファイル"""

    base: str = "./data"
    train_csv: str = f"{base}/train.csv"
    features_csv: str = f"{base}/features.csv"
    pt_notes_csv: str = f"{base}/patient_notes.csv"
    test_csv: str = f"{base}/test.csv"
    submit_csv: str = f"{base}/sample_submission.csv"


@dataclass
class Output:
    """出力ディレクトリ・ファイル"""

    base: str = "./weights"
    config: str = f"{base}/config.pth"
    oof_df: str = f"{base}/oof_df.pkl"
    tokenizer: str = f"{base}/tokenizer/"
    submit_csv: str = "./submission.csv"
