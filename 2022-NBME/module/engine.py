import time
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from .config import Config
from .utils import AverageMeter, get_logger, timeSince

logger = get_logger(__name__)


def train_fn(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    device: torch._C.device,
) -> float:
    """訓練用関数

    Args:
        dataloader (DataLoader): データローダー
        model (nn.Module): モデル
        criterion (nn.modules.loss._Loss): 損失関数
        optimizer (optim.Optimizer): 最適化アルゴリズム
        scheduler (optim.lr_scheduler._LRScheduler): 学習率スケジューラ
        device (torch._C.device): 使用するデバイス. torch.device("cpu") or torch.device("cuda")

    Returns:
        float: Epochの損失の平均値
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=Config.apex)
    losses = AverageMeter()
    start: float = time.time()
    global_step: int = 0
    for step, (inputs, labels) in enumerate(dataloader):
        for key, val in inputs.items():
            inputs[key] = val.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=Config.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        losses.update(loss.item(), batch_size)
        if Config.optimizer.grad_accum_steps > 1:
            loss = loss / Config.optimizer.grad_accum_steps
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), Config.optimizer.max_grad_norm
        )
        if (step + 1) % Config.optimizer.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if Config.optimizer.batch_scheduler:
                scheduler.step()
        if step % Config.log_freq == 0 or step == (len(dataloader) - 1):
            logger.info(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f} "
                "LR: {lr:.8f} ".format(
                    epoch + 1,
                    step,
                    len(dataloader),
                    remain=timeSince(start, float(step + 1) / len(dataloader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],  # type: ignore
                )
            )
    return losses.avg


def valid_fn(
    dataloader: DataLoader, model: nn.Module, criterion: nn.modules.loss._Loss, device: torch._C.device  # type: ignore
) -> Tuple[float, np.ndarray]:
    """検証用関数

    Args:
        dataloader (DataLoader): データローダー
        model (nn.Module): モデル
        criterion (nn.modules.loss._Loss): 損失関数
        device (torch._C.device): 使用するデバイス. torch.device("cpu") or torch.device("cuda")

    Returns:
        float: Epochの損失の平均値
        np.ndarray: 検証データの予測
    """
    model.eval()
    losses = AverageMeter()
    preds: List[np.ndarray] = []
    start: float = time.time()
    for step, (inputs, labels) in enumerate(dataloader):
        for key, val in inputs.items():
            inputs[key] = val.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        losses.update(loss.item(), batch_size)
        if Config.optimizer.grad_accum_steps > 1:
            loss = loss / Config.optimizer.grad_accum_steps
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        if step % Config.log_freq == 0 or step == (len(dataloader) - 1):
            logger.info(
                "EVAL: [{0}/{1}] "
                "Elapsed: {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(dataloader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(dataloader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference_fn(
    dataloder: DataLoader, model: nn.Module, device: torch._C.device
) -> np.ndarray:
    """予測用関数

    Args:
        dataloder (DataLoader): データローダー
        model (nn.Module): モデル
        device (torch._C.device): デバイス.

    Returns:
        np.ndarray: 予測値の行列
    """
    model.eval()
    model.to(device)
    preds = []
    progress = tqdm(dataloder, total=len(dataloder))
    for inputs in progress:
        for key, val in inputs.items():
            inputs[key] = val.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions


def inference_fn_fast(
    dataloder: DataLoader, model: nn.Module, device: torch._C.device
) -> np.ndarray:
    """予測用関数

    Args:
        dataloder (DataLoader): データローダー
        model (nn.Module): モデル
        device (torch._C.device): デバイス.

    Returns:
        np.ndarray: 予測値の行列
    """
    model.eval()
    model.to(device)
    preds = []
    progress = tqdm(dataloder, total=len(dataloder))
    for inputs in progress:
        bs = len(inputs["input_ids"])
        pred_w_pad = np.zeros((bs, Config.model.max_length, 1))
        for key, val in inputs.items():
            inputs[key] = val.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        y_preds = y_preds.sigmoid().to("cpu").numpy()
        pred_w_pad[:, : y_preds.shape[1]] = y_preds
        preds.append(pred_w_pad)
    predictions = np.concatenate(preds)
    return predictions
