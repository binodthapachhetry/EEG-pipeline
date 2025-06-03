#!/usr/bin/env python3
# Subscribe to EEGWin LSL stream, run DL inference, publish state estimate markers.
import argparse, json, numpy as np, torch, time
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
from dl_models import load_pretrained

def main():
    p = argparse.ArgumentParser(description="Real-time EEG state classifier.")
    p.add_argument("--model", choices=["cnn","tcn","transformer"], default="cnn")
    p.add_argument("--ckpt",  required=True, help="Path to .pt weights")
    p.add_argument("--n_classes", type=int, default=4)
    args = p.parse_args()

    print("Resolving EEGWin stream…")
    inlet = StreamInlet(resolve_stream("name","EEGWin",1,5)[0])
    n_ch   = inlet.info().channel_count()
    sfreq  = inlet.info().nominal_srate()
    win_samples = int(inlet.info().desc().child_value("window_samples"))

    print(f"Model: {args.model} | channels={n_ch} | win={win_samples}")
    model = load_pretrained(args.model, args.ckpt, n_ch, args.n_classes)

    # Outlet for predictions
    info = StreamInfo("EEGState","Markers",1,0,"string","state-stream")
    outlet = StreamOutlet(info)

    buf = np.zeros((win_samples,n_ch),dtype=np.float32)
    while True:
        inlet.pull_chunk(buf, max_samples=win_samples)  # blocking until full window
        x = torch.from_numpy(buf.T).unsqueeze(0)        # shape (1,c,t)
        with torch.no_grad():
            pred = torch.softmax(model(x),-1).cpu().numpy()[0]
        label = int(pred.argmax())
        outlet.push_sample([json.dumps({"label":label,"probs":pred.tolist()})], pushthrough=True)
        time.sleep(win_samples/sfreq)  # pace loop

if __name__ == "__main__":
    main()
