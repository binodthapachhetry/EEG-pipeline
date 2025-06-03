#!/usr/bin/env python3
# Generate realistic synthetic EEG using MNE; fall back to sinusoid+noise when TVB unavailable.
import argparse, logging, time, numpy as np, mne
from typing import List
try:
    from tvb.simulator.lab import simulator, models, coupling; TVB_OK = True
except ImportError:
    TVB_OK = False
from lsl_stream import LSLWindowStreamer

def _tvb_ts(n_ch:int, sf:float, dur:float)->np.ndarray:
    sim = simulator.Simulator(model=models.Generic2dOscillator(),
                              coupling=coupling.Linear(a=0.003),
                              simulation_length=dur*1e3,
                              sampling_step_size=1e3/sf).configure()
    _, data, _ = sim.run()
    return data[:n_ch,:].astype(np.float32)

def gen_raw(ch_names:List[str], sf:float, dur:float)->mne.io.RawArray:
    n_samp = int(sf*dur)
    if TVB_OK:
        data = _tvb_ts(len(ch_names), sf, dur)
    else:
        t = np.arange(n_samp)/sf
        freqs = np.random.uniform(6,15,len(ch_names))
        data = 1e-5*np.vstack([np.sin(2*np.pi*f*t) for f in freqs]).astype(np.float32)
        data += 2e-6*np.random.randn(len(ch_names), n_samp)
    info = mne.create_info(ch_names, sf, ch_types=['eeg']*len(ch_names))
    return mne.io.RawArray(data, info)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration",type=float,default=60)
    parser.add_argument("--sfreq",type=float,default=250)
    parser.add_argument("--channels",type=int,default=8)
    parser.add_argument("--window",type=float,default=7.0)
    args = parser.parse_args()

    ch = [f"EEG{i}" for i in range(args.channels)]
    raw = gen_raw(ch, args.sfreq, args.duration)
    win_samp = int(args.window*args.sfreq)
    streamer = LSLWindowStreamer(ch, args.sfreq, win_samp)

    for t0 in np.arange(0, args.duration-args.window+1e-3, args.window):
        seg = raw.copy().crop(tmin=t0, tmax=t0+args.window, include_tmax=False)
        streamer.push_window(seg.get_data(), {}, [])
        time.sleep(args.window)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
    main()

__all__=["gen_raw"]
