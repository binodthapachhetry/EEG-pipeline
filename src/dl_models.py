#!/usr/bin/env python3
"""
Light-weight EEG deep-learning model definitions for real-time inference.
Includes:
  • ShallowConvNet (CNN, Braindecode style)
  • TemporalConvNet  (TCN)
  • EEG-Transformer (token-wise self-attention)
`load_pretrained(model_name, ckpt_path)` returns a `.eval()` model on CPU/GPU.
"""
from pathlib import Path
from typing import Literal
import torch, torch.nn as nn
from einops.layers.torch import Rearrange

# ---------- Model definitions ----------
class ShallowConvNet(nn.Module):
    def __init__(self, n_ch:int, n_classes:int):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange('b c t -> b 1 c t'),
            nn.Conv2d(1, 40, (1, 25), padding=(0,12), bias=False),
            nn.Conv2d(40, 40, (n_ch, 1), bias=False),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 75)), nn.Dropout(0.5),
            Rearrange('b f 1 t -> b (f t)'),
            nn.Linear(40* ((1000//75)+1), n_classes)
        )
    def forward(self,x): return self.net(x)

class SimpleTCN(nn.Module):
    def __init__(self, n_ch:int, n_classes:int, levels:int=4, k:int=8):
        super().__init__()
        layers=[]
        in_ch=n_ch
        for i in range(levels):
            layers.append(nn.Conv1d(in_ch, k, 3, padding=2**i, dilation=2**i))
            layers.append(nn.ReLU()); layers.append(nn.BatchNorm1d(k))
            in_ch=k
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(k, n_classes)
    def forward(self,x):               # x: (b,c,t)
        y = self.tcn(x); y = y.mean(-1) # global avg-pool
        return self.fc(y)

class EEGTransformer(nn.Module):
    def __init__(self, n_ch:int, n_classes:int, d_model:int=64, n_heads:int=4):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1,1024,d_model))
        self.proj = nn.Linear(n_ch, d_model)
        encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*2, batch_first=True),
            num_layers=4
        )
        self.enc = encoder
        self.head= nn.Linear(d_model, n_classes)
    def forward(self,x):               # (b,c,t) -> (b,t,c)
        x = x.permute(0,2,1)
        x = self.proj(x) + self.pos[:,:x.size(1)]
        z = self.enc(x).mean(1)
        return self.head(z)

# ---------- Loader ----------
def load_pretrained(model:Literal['cnn','tcn','transformer'], ckpt:str, n_ch:int, n_classes:int):
    model_map={'cnn':ShallowConvNet,'tcn':SimpleTCN,'transformer':EEGTransformer}
    m = model_map[model](n_ch, n_classes)
    ckpt_path = Path(ckpt)
    if ckpt_path.is_file():
        m.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    m.eval()
    return m.cuda() if torch.cuda.is_available() else m

__all__=['load_pretrained','ShallowConvNet','SimpleTCN','EEGTransformer']
