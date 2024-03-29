from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .config import Config


class CustomModel(nn.Module):
    def __init__(self, config_path: Optional[str] = None, pretrained: bool = False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                Config.model.name, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(
                Config.model.name, config=self.config
            )
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(Config.model.dropout_ratio)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output
