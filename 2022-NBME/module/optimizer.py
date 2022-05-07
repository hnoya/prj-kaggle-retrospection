from typing import List

from torch import nn, optim
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .config import Config, Optimizers, Schedulers


def get_optimizer_params(
    model: nn.Module, encoder_lr: float, decoder_lr: float, weight_decay: float = 0.0
) -> List[dict]:
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_paramters = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()  # type: ignore
                if not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.model.named_parameters()  # type: ignore
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_paramters


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.optimizer.encoder_lr,
        decoder_lr=Config.optimizer.decoder_lr,
        weight_decay=Config.optimizer.weight_decay,
    )
    if Config.optimizer.name == Optimizers.AdamW:
        optimizer = optim.AdamW(
            optimizer_parameters,
            lr=Config.optimizer.encoder_lr,
            eps=Config.optimizer.eps,
            betas=Config.optimizer.betas,
        )
    else:
        assert False, "Invalid optimzier name."
    return optimizer


def get_scheduler(
    optimizer: optim.Optimizer, num_train_steps: int
) -> optim.lr_scheduler._LRScheduler:
    """Optimizer Schedulerを取得する

    Args:
        optimizer (optim.Optimizer): optimizer
        num_train_steps (int): 訓練時の全Epochの総Step数

    Returns:
        optim.lr_scheduler._LRScheduler: Optimizer Scheduler
    """
    if Config.optimizer.scheduler == Schedulers.linear:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.optimizer.num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif Config.optimizer.scheduler == Schedulers.cosine:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.optimizer.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=Config.optimizer.num_cycles,
        )
    else:
        assert False, "Invalid Sheduler."
    return scheduler
