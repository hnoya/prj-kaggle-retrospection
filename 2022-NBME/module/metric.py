from typing import List, Optional

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(preds: List[np.ndarray], truths: List[np.ndarray]) -> float:
    """マクロF1評価

    Args:
        preds (List[np.ndarray]): 文字単位の2値予測行列のリスト
        truths (List[np.ndarray]): 文字単位の2値ラベル行列のリスト

    Returns:
        float: マクロF1評価
    """
    preds_arr: np.ndarray = np.concatenate(preds)
    truths_arr: np.ndarray = np.concatenate(truths)
    return f1_score(truths_arr, preds_arr)


def spans_to_binary(
    spans: List[List[int]], max_lenght: Optional[int] = None
) -> np.ndarray:
    """スパン行列から2値行列に変換する

    Args:
        spans (List[List[int]]): [start, end]からなるスパン行列
        max_lenght (Optional[int], optional): 2値行列の最大長. Defaults to None.

    Returns:
        np.ndarray: 2値行列
    """
    max_length = np.max(spans) if max_lenght is None else max_lenght
    binary_arr = np.zeros(max_length)
    for start, end in spans:
        binary_arr[start:end] = 1
    return binary_arr


def span_micro_f1(
    preds_span: List[List[List[int]]], truths_span: List[List[List[int]]]
) -> float:
    """スパン行列からマクロF1評価を行う

    Args:
        preds_span (List[List[List[int]]]): データ全体の予測したスパン行列
        truths_span (List[List[List[int]]]): データ全体の実際のスパン行列

    Returns:
        float: 評価したマクロF1
    """
    binary_preds = []
    binary_truths = []
    for pred_span, truth_span in zip(preds_span, truths_span):
        if not len(pred_span) and not len(truth_span):
            continue
        max_length = max(
            np.max(pred_span) if len(pred_span) else 0,
            np.max(truth_span) if len(truth_span) else 0,
        )
        binary_preds.append(spans_to_binary(pred_span, max_length))
        binary_truths.append(spans_to_binary(truth_span, max_length))
    return macro_f1(binary_preds, binary_truths)


def get_score(labels: List[List[List[int]]], preds: List[List[List[int]]]) -> float:
    return span_micro_f1(labels, preds)
