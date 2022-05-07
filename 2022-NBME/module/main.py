import gc
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config, Input, Output, train_col
from .dataset import TestDatasetFast, TrainDataset, load_test_df_fast, load_train_df
from .engine import inference_fn_fast, train_fn, valid_fn
from .metric import get_score
from .model import CustomModel
from .optimizer import get_optimizer, get_scheduler
from .submit import (
    create_labels_for_scoring,
    get_char_probs,
    get_predictions_for_submit,
    labels_to_sub,
    post_process_spaces,
)
from .utils import get_logger

logger = get_logger(__name__)


def log_score(oof_df: pd.DataFrame):
    labels = create_labels_for_scoring(oof_df)
    preds = get_predictions_for_submit(
        texts=oof_df[train_col.pn_history].values,
        predictions=oof_df[[i for i in range(Config.model.max_length)]].values,
    )
    score = get_score(labels, preds)
    logger.info(f"Score: {score:<.4f}")


def train_one_fold(df: pd.DataFrame, fold: int) -> pd.DataFrame:
    """1Fold学習する

    Args:
        df (pd.DataFrame): 学習用データフレーム
        fold (int): 学習するfold番号

    Returns:
        pd.DataFrame: 推論結果が入った検証用データフレーム
    """
    train = df[df["fold"] != fold].reset_index(drop=True)
    valid = df[df["fold"] == fold].reset_index(drop=True)
    valid_texts = valid[train_col.pn_history].values
    valid_labels = create_labels_for_scoring(valid)

    train_dataloader = DataLoader(
        TrainDataset(train),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        TrainDataset(valid),
        batch_size=Config.batch_size * 3,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = CustomModel(config_path=None, pretrained=True)
    torch.save(model.config, Output.config)
    model.to(Config.device)
    optimizer = get_optimizer(model)
    num_train_steps = int(len(train) / Config.batch_size * Config.epochs)
    scheduler = get_scheduler(optimizer, num_train_steps)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_score = -1.0
    for epoch in range(Config.epochs):
        start_time = time.time()
        avg_loss = train_fn(
            train_dataloader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            Config.device,
        )
        avg_val_loss, predictions = valid_fn(
            valid_dataloader, model, criterion, Config.device
        )
        preds = get_predictions_for_submit(valid_texts, predictions)
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        logger.info(f"Epoch {epoch+1} - Score: {score:.4f}")

        if best_score < score:
            best_score = score
            logger.info(f"Epoch {epoch+1} - Save Best Loss: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                f"{Output.base}/{Config.model.name.replace('/', '-')}_fold{fold}_best.pth",
            )

    predictions = torch.load(
        f"{Output.base}/{Config.model.name.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
    )["predictions"]
    valid[[i for i in range(Config.model.max_length)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    return valid


def train(
    train_path: str = Input.train_csv,
    features_path: str = Input.features_csv,
    pt_notes_path: str = Input.pt_notes_csv,
):
    """訓練用API

    Args:
        train_path (str, optional): 訓練用データセットcsvパス. Defaults to Input.train_csv.
        features_path (str, optional): コンテキストデータセットcsvパス. Defaults to Input.features_csv.
        pt_notes_path (str, optional): 患者ノートデータセットのcsvパス. Defaults to Input.pt_notes_csv.
    """
    oof_df = pd.DataFrame()
    df = load_train_df(
        train_path=train_path, features_path=features_path, pt_notes_path=pt_notes_path
    )
    for fold in range(Config.n_fold):
        if fold in Config.trn_fold:
            _oof_df = train_one_fold(df, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            logger.info(f"========== fold: {fold} result ==========")
            log_score(_oof_df)
    oof_df = oof_df.reset_index(drop=True)
    logger.info("========== CV ==========")
    log_score(oof_df)
    oof_df.to_pickle(Output.oof_df)


def test_folds(
    sort_df: pd.DataFrame, df: pd.DataFrame, length_sorted_idx: np.ndarray
) -> np.ndarray:
    """Configモデルについて予測する

    Args:
        sort_df (pd.DataFrame): ソート済みテストデータフレーム
        df (pd.DataFrame): テストデータフレーム
        length_sorted_idx (np.ndarray): ソート用インデックス行列

    Returns:
        np.ndarray: 予測結果
    """
    test_dataset = TestDatasetFast(sort_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    predictions = []
    for fold in Config.trn_fold:
        model = CustomModel(config_path=Output.config, pretrained=False)
        state = torch.load(
            f"{Output.base}/{Config.model.name.replace('/', '-')}_fold{fold}_best.pth"
        )
        model.load_state_dict(state["model"])
        prediction = inference_fn_fast(test_loader, model, Config.device)
        prediction = prediction.reshape((len(sort_df), Config.model.max_length))
        prediction = prediction[np.argsort(length_sorted_idx)]
        char_probs = get_char_probs(
            df[train_col.pn_history].values, prediction, Config.model.tokenizer
        )
        predictions.append(char_probs)
        del model, state, prediction, char_probs
        gc.collect()
        torch.cuda.empty_cache()
    predictions_arr = np.mean(predictions, axis=0)
    return predictions_arr


def test(
    test_path: str = Input.test_csv,
    features_path: str = Input.features_csv,
    pt_notes_path: str = Input.pt_notes_csv,
) -> List[str]:
    """予測用API

    Args:
        test_path (str, optional): 予測用データセットcsvパス. Defaults to Input.test_csv.
        features_path (str, optional): コンテキストデータセットcsvパス. Defaults to Input.features_csv.
        pt_notes_path (str, optional): 患者ノートデータセットのcsvパス. Defaults to Input.pt_notes_csv.

    Returns:
        List[str]: 予測結果
    """
    sort_df, df, length_sorted_idx = load_test_df_fast(
        test_path, features_path, pt_notes_path
    )
    predictions = test_folds(sort_df, df, length_sorted_idx)
    df["preds"] = [prediction for prediction in predictions]
    df["preds"] = df["preds"].apply(lambda x: x > 0.5)
    df["preds_pp"] = df.apply(
        lambda x: post_process_spaces(x["preds"], x["clean_text"]), 1
    )
    location_list = labels_to_sub(df["preds_pp"].values)
    submission = pd.read_csv(Input.submit_csv)
    submission[train_col.location] = location_list
    submission[["id", "location"]].to_csv(Output.submit_csv, index=False)
    return location_list
