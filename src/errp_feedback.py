#!/usr/bin/env python3
# Online ErrP detector that tunes the LinUCB bandit using implicit feedback.
import numpy as np, torch
from pylsl import StreamInlet, resolve_stream
from rl_bandit import LinUCBBandit

class SimpleErrPNet(torch.nn.Module):
    def __init__(self, n_ch:int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_ch,16,7,padding=3), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
            torch.nn.Linear(16,2)
        )
    def forward(self,x): return self.net(x)

def online_loop():
    eeg_in = StreamInlet(resolve_stream("name","EEGWin",1,5)[0])
    state_in = StreamInlet(resolve_stream("name","EEGState",1,5)[0]) # predictions

    n_actions = 3
    bandit = LinUCBBandit(n_actions, n_features=10)  # feature dim placeholder
    errp_net = SimpleErrPNet(eeg_in.info().channel_count()).eval()

    eeg_buf = np.zeros((eeg_in.info().channel_count(), 256))  # 1-s ErrP snippet

    while True:
        # 1) wait for model decision marker
        _,ts = state_in.pull_sample()
        # 2) grab post-stimulus 1-s EEG for ErrP detection
        eeg_in.pull_chunk(eeg_buf.T, max_samples=256)
        with torch.no_grad():
            prob_err = torch.softmax(errp_net(torch.from_numpy(eeg_buf).unsqueeze(0)), -1)[0,1].item()
        reward = 1.0 - prob_err  # large reward when no ErrP
        # dummy context for now
        ctx = np.random.randn(10)
        bandit.update(ctx, action=0, reward=reward)

if __name__=="__main__":
    online_loop()
