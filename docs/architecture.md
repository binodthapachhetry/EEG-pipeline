# NeuroFeedbackLab – System Architecture

## 1. High-level Overview
```
                ┌───────────┐      EDF/FIF     ┌────────────┐
  Offline path  │ read_edf  ├───────────────► │ preprocess │
                └───────────┘                 └────────────┘
                      ▲                              │
                      │                              │  (windows + features)
     Synthetic path   │                              ▼
                ┌──────────────┐   LSL  ┌──────────────────────┐
                │ simulate_eeg │ ─────► │  LSLWindowStreamer   │
                └──────────────┘        └──────────────────────┘
                                             │         ▲
                                             │         │
                            LSL              ▼         │ LSL
                ┌─────────────────────┐   ┌─────────────────────┐
                │ realtime_classifier │   │  EegStreamProcessor │
                └─────────────────────┘   └─────────────────────┘
                         │                         │
                         │ ErrP markers            │ Spark Sink
                         ▼                         ▼
                ┌─────────────────────┐   ┌─────────────────────┐
                │   errp_feedback     │   │ External Analytics  │
                └─────────────────────┘   └─────────────────────┘
```

### Component Responsibilities
| Layer | Module(s) | Role |
|-------|-----------|------|
| Data Acquisition | **read_edf.py** | Clean raw EDF, save as FIF |
| Simulation | **simulate_eeg.py** | Produce realistic EEG, stream via LSL |
| Pre-processing | **preprocess_eeg.py**, **lsl_stream.py** | Sliding-window filtering + feature extraction; publish `EEGWin` stream |
| Online DL Inference | **dl_models.py**, **realtime_classifier.py** | Load pretrained CNN/TCN/Transformer, publish `EEGState` stream |
| Reinforcement Learning | **rl_bandit.py**, **errp_feedback.py** | Detect ErrP → convert to reward → LinUCB update |
| Large-scale Analytics | **scala-streaming/** | Ingest `EEGWin` into Spark for clustering, MLlib, storage |
| Interchange | **docs/EBS_FORMAT.md** | Compact binary archive; future writer/reader pending |

## 2. Data Flow
1. **Acquisition** – either `read_edf.py` (offline) or `simulate_eeg.py` (online synthetic) produces multichannel EEG.  
2. **Pre-processing** – `preprocess_eeg.py` (or `simulate_eeg.py` for synthetic) segments into windows, extracts spectral features and streams via LSL (`EEGWin`).  
3. **Inference** – `realtime_classifier.py` consumes `EEGWin`, runs a deep-learning model, and emits brain-state markers on `EEGState`.  
4. **Adaptation** – `errp_feedback.py` listens for `EEGState`, captures post-decision EEG to detect ErrPs; the probability of an error becomes a reward fed into a LinUCB bandit that can later pick interventions.  
5. **Analytics / Storage** – `EegStreamProcessor.scala` ingests `EEGWin` in Spark Structured Streaming to support cluster-scale analytics or writing to durable stores (Delta Lake, Kafka, Parquet, etc.).  

## 3. Technology Stack
• Python 3.10 ‑ MNE, PyTorch, Braindecode, TVB, pylsl  
• Scala 2.13 + Spark 3.5 (Structured Streaming, MLlib) + liblsl-java  
• LSL bridges all online components; JSON markers carry metadata.  

## 4. Remaining / Planned Work
1. **Model Training & Management** – scripts for training DL models and exporting `.pt` checkpoints; on-demand model hot-swap.  
2. **Multi-modal Fusion** – extend `realtime_classifier.py` to accept additional LSL streams (e.g., HRV, GSR) and feed a fusion network.  
3. **Closed-loop Actions** – integrate the LinUCB bandit with an intervention delivery layer (audio/visual/tACS).  
4. **EBS Writer/Reader** – implement I/O modules per `docs/EBS_FORMAT.md`; enable Spark to archive and replay streams.  
5. **Unit & Integration Tests** – automated tests for each module; CI workflow.  
6. **Deployment** – Docker images, Kubernetes Helm charts, GPU scheduling.  

## 5. Runtime Scenarios
### A. Offline Experiment Re-processing  
1. Convert EDF → FIF.  
2. Run `preprocess_eeg.py` (no LSL) to produce features for classical ML or DL fine-tuning.  
### B. Live Demo Without Hardware  
1. `simulate_eeg.py --stream`.  
2. `realtime_classifier.py --ckpt my_model.pt`.  
3. Optionally `errp_feedback.py` and `sbt run` for analytics.  
### C. Full Closed-loop Session  
1. Real headset streams raw LSL (future).  
2. Replace step 1 with a live‐capture module; rest of pipeline remains unchanged due to LSL abstraction.  

---
© NeuroFeedbackLab – architecture v0.2
