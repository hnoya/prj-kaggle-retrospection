import ast
import itertools
from typing import List

import numpy as np
import pandas as pd
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .config import Config, train_col


def create_labels_for_scoring(df: pd.DataFrame) -> List[List[List[int]]]:
    """スコア計算のためのラベル付を行う

    Args:
        df (pd.DataFrame): データフレーム

    Returns:
        List[List[List[int]]]: 推論結果のリスト
    """
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df["_location_for_create_labels"] = [ast.literal_eval("[]")] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, train_col.location]
        if lst:
            new_lst = ";".join(lst)
            df.loc[i, "_location_for_create_labels"] = ast.literal_eval(
                f'[["{new_lst}"]]'
            )
    # create labels
    truths: List[List[List[int]]] = []
    for location_list in df["_location_for_create_labels"].values:
        truth: List[List[int]] = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(";")]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths


def get_char_probs(
    texts: List[str], predictions: np.ndarray, tokenizer: PreTrainedTokenizerBase
) -> List[np.ndarray]:
    """テキスト行列とtoken単位の予測行列を使い、文字単位の予測行列を取得する

    Args:
        texts (List[str]): テキスト行列
        predictions (np.ndarray): token単位の予測確率の行列
        tokenizer (PreTrainedTokenizerBase): 使用したtokenizer

    Returns:
        List[np.ndarray]: 文字単位の予測行列
    """
    results: List[np.ndarray] = [np.zeros(len(text)) for text in texts]
    for idx, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded: BatchEncoding = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        for offset_mapping, pred in zip(encoded["offset_mapping"], prediction):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[idx][start:end] = pred
    return results


def get_results(
    char_probs: List[np.ndarray], threshold: float = 0.5
) -> List[List[str]]:
    """予測結果を取得する

    Args:
        char_probs (List[np.ndarray]): 文字単位の予測結果行列
        threshold (float, optional): 正とみなす確率のしきい値. Defaults to 0.5.

    Returns:
        List[List[str]]: 予測結果
    """
    results: List[List[str]] = []
    for char_prob in char_probs:
        result_arr: np.ndarray = np.where(char_prob >= threshold)[0] + 1
        result_idxs: List[List[int]] = [
            list(g)
            for _, g in itertools.groupby(
                result_arr, key=lambda n, c=itertools.count(): n - next(c)  # type: ignore
            )
        ]
        result: List[str] = [f"{min(r)} {max(r)}" for r in result_idxs]
        results.append(result)
    return results


def get_predictions(results: List[List[str]]) -> List[List[List[int]]]:
    """予測行列を取得する

    Args:
        results (List[List[str]]): 結果のスパン行列

    Returns:
        List[List[List[int]]]: 予測のスパン行列
    """
    predictions: List[List[List[int]]] = []
    for result in results:
        prediction: List[List[int]] = []
        if result != "":
            for loc in [s.split() for s in result.split(";")]:  # type: ignore # TODO
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions


def get_predictions_for_submit(
    texts: List[str], predictions: np.ndarray
) -> List[List[List[int]]]:
    """提出形式の予測結果を取得する

    Args:
        texts (List[str]): テキストのリスト
        predictions (np.ndarray): 予測結果

    Returns:
        List[List[List[int]]]: 提出形式の予測結果
    """
    char_probs = get_char_probs(texts, predictions, Config.model.tokenizer)
    results = get_results(char_probs, threshold=0.5)
    preds = get_predictions(results)
    return preds


def post_process_spaces(target: np.ndarray, text: List[str]) -> np.ndarray:
    """予測後のスペースの後処理

    Args:
        target (np.ndarray): 予測した箇所のリスト
        text (List[str]): テキスト

    Returns:
        np.ndarray: 後処理後のリスト
    """
    target = np.copy(target)

    if len(text) > len(target):
        padding = np.zeros(len(text) - len(target))
        target = np.concatenate([target, padding])
    else:
        target = target[: len(text)]

    if text[0] == " ":
        target[0] = 0
    if text[-1] == " ":
        target[-1] = 0

    for i in range(1, len(text) - 1):
        if text[i] == " ":
            if target[i] and not target[i - 1]:  # space before
                target[i] = 0

            if target[i] and not target[i + 1]:  # space after
                target[i] = 0

            if target[i - 1] and target[i + 1]:
                target[i] = 1
    return target


def labels_to_sub(labels: np.ndarray) -> List[str]:
    """予測したスパンを提出形式のリストに変形する

    Args:
        labels (np.ndarray): 予測したスパン

    Returns:
        List[str]: 提出形式のリスト
    """
    all_spans = []
    for label in labels:
        indices = np.where(label > 0)[0]
        indices_grouped = [
            list(g)
            for _, g in itertools.groupby(
                indices, key=lambda n, c=itertools.count(): n - next(c)  # type: ignore
            )
        ]

        spans = [f"{min(r)} {max(r) + 1}" for r in indices_grouped]
        all_spans.append(";".join(spans))
    return all_spans
