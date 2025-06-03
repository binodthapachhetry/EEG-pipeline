#!/usr/bin/env python3
# Stream pre-processed EEG windows via LabStreamingLayer.
from pylsl import StreamInfo, StreamOutlet
import numpy as np, time, json

class LSLWindowStreamer:
    def __init__(self, ch_names, sfreq, win_samples):
        info = StreamInfo('EEGWin','EEG',len(ch_names),sfreq,'float32','eeg-win-01')
        info.desc().append_child_value('manufacturer','NeuroFeedbackLab')
        info.desc().append_child_value('window_samples',str(win_samples))
        self.outlet = StreamOutlet(info,chunk_size=win_samples)
        self.ch_names = ch_names

    def push_window(self, window_np: np.ndarray, features: dict, stages: list[str]):
        """
        window_np: shape (channels, samples)
        Features & stages are serialised into a JSON timestamped marker stream.
        """
        # LSL expects samples as (n_samples, n_channels)
        self.outlet.push_chunk(window_np.T.astype(np.float32))
        # Push meta-marker
        self.outlet.push_sample([json.dumps({'features':features,'stages':stages})], pushthrough=True)

__all__=['LSLWindowStreamer']
