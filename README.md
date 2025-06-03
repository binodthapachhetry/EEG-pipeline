# NeuroFeedbackLab – Real-Time EEG Analytics & Adaptive Intervention

## 1 . Project Purpose
This repository turns raw or simulated EEG into **real-time, AI-driven feedback**.  
Pipeline highlights  
1. Acquire EEG (EDF file _or_ live TVB/MNE simulation).  
2. Slide windows, filter, extract spectral features, broadcast via **LSL** (`EEGWin`).  
3. Run lightweight **CNN/TCN/Transformer** models to classify brain-state; publish markers (`EEGState`).  
4. Detect Error-Related Potentials (ErrP) and update a **LinUCB** bandit for closed-loop personalisation.  
5. Optionally ingest raw windows into **Spark Structured Streaming** for cluster-scale analytics.  

## 2 . Repository Layout
| Path | Role |
|------|------|
| `src/read_edf.py` | offline EDF → FIF converter |
| `src/preprocess_eeg.py` | windowed filtering & feature extraction (+ optional LSL out) |
| `src/simulate_eeg.py` | realistic EEG generator (TVB or sine + noise) + LSL out |
| `src/lsl_stream.py` | helper for publishing windows to LSL |
| `src/dl_models.py` | ShallowConvNet, TCN, Transformer & loader |
| `src/realtime_classifier.py` | subscribes to `EEGWin`, outputs `EEGState` |
| `src/rl_bandit.py` | LinUCB contextual bandit core |
| `src/errp_feedback.py` | ErrP detector + bandit reward loop |
| `scala-streaming/` | Spark ingestion (`EegStreamProcessor.scala`) & `build.sbt` |
| `docs/EBS_FORMAT.md` | future binary archive spec |

## 3 . Quick Start
### 3.1 Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional: TVB simulator
pip install tvb-library
```
For Spark side:  
```bash
cd scala-streaming
sbt assembly   # or sbt run for local test
```

### 3.2 Run an End-to-End Demo (no hardware)
```bash
# Terminal 1 – simulate EEG & stream
python src/simulate_eeg.py --duration 30 --sfreq 250 --channels 8 --window 7

# Terminal 2 – classify brain state
python src/realtime_classifier.py --model cnn --ckpt weights/my_cnn.pt --n_classes 4

# Terminal 3 – ErrP-driven adaptation
python src/errp_feedback.py
```
View streams with LabRecorder or run the Spark job:
```bash
cd scala-streaming && sbt run
```

### 3.3 Offline Pre-processing
```bash
python src/read_edf.py data/raw.edf data/clean.fif
python src/preprocess_eeg.py data/clean.fif --window_sec 7 --overlap_sec 2 --stream_lsl
```

## 4 . Model Training & Checkpoints
`dl_models.py` holds small architectures.  
Train externally (e.g., Braindecode) and export to `.pt`; `realtime_classifier.py` will auto-load.

## 5 . Extending the System
• **Add new sensors** – publish additional LSL streams and modify `realtime_classifier.py` for multi-modal fusion.  
• **Swap RL algorithm** – implement another agent in `src/rl_bandit.py`; wire it in `errp_feedback.py`.  
• **Large LLM interventions** – connect `errp_feedback.py` rewards to a backend service that chooses/personalises prompts.  
• **Persist streams** – implement EBS writer/reader per `docs/EBS_FORMAT.md` or use Spark sinks (Parquet, Kafka, Delta).  

## 6 . Testing
```bash
ruff src
mypy src
pytest          # (when tests are added)
```

## 7 . License
Distributed under the MIT License (see `LICENSE`).

---
© NeuroFeedbackLab 2024
