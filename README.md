# psg-eeg-processor – Real-Time EEG Processing & Neurofeedback Pipeline                                                              
                                                                                                                                                                                              
 End-to-end toolkit for streaming, classifying, and adapting EEG data in real time – from raw PSG/EDF to deep learning and            
 reinforcement learning–driven neurofeedback.                                                                                         
                                                                                                           
 ---                                                                                                                                  
                                                                                                                                      
 ## Table of Contents                                                                                                                 
                                                                                                                                      
 - [1. Quick Start](#1-quick-start)                                                                                                   
 - [2. Architecture Overview](#2-architecture-overview)                                                                               
 - [3. Component Walk-through](#3-component-walk-through)                                                                             
 - [4. Example Workflows](#4-example-workflows)                                                                                       
 - [5. Development & Testing](#5-development--testing)                                                                                
 - [6. FAQ / Troubleshooting](#6-faq--troubleshooting)                                                                                
 - [7. License](#7-license)                                                                                                           
                                                                                                                                      
 ---                                                                                                                                  
                                                                                                                                      
 ## 1. Quick Start                                                                                                                    
                                                                                                                                      
 **Environment Setup**                                                                                                                
 ```bash                                                                                                                              
 python -m venv venv                                                                                                                  
 source venv/bin/activate                                                                                                             
 pip install -r requirements.txt                                                                                                      
 # (Optional) For TVB-based simulation:                                                                                               
 pip install tvb-library                                                                                                              
 ```                                                                                                                                  
 **Scala/Spark (optional, for streaming analytics):**                                                                                 
 ```bash                                                                                                                              
 cd scala-streaming                                                                                                                   
 sbt assembly   # or: sbt run                                                                                                         
 ```                                                                                                                                  
                                                                                                                                      
 **Demo: Real-Time Pipeline in 3 Steps**                                                                                              
 1. **Simulate EEG & stream via LSL**                                                                                                 
    ```bash                                                                                                                           
    python src/simulate_eeg.py --duration 30 --sfreq 250 --channels 8 --window 7                                                      
    ```                                                                                                                               
 2. **Classify brain state (DL inference)**                                                                                           
    ```bash                                                                                                                           
    python src/realtime_classifier.py --model cnn --ckpt weights/my_cnn.pt --n_classes 4                                              
    ```                                                                                                                               
 3. **ErrP-driven adaptation (RL bandit)**                                                                                            
    ```bash                                                                                                                           
    python src/errp_feedback.py                                                                                                       
    ```                                                                                                                               
                                                                                                                                      
 ---                                                                                                                                  
                                                                                                                                      
 ## 2. Architecture Overview                                                                                                          
                                                                                                                                      
 ```                                                                                                                                  
 [EDF/FIF] → [preprocess_eeg.py] → [lsl_stream.py] ⇄ [realtime_classifier.py] ⇄ [errp_feedback.py]                                    
    ↑             ↑                     ↑                    ↑                                                                        
 [read_edf.py] [simulate_eeg.py]   [Spark/Scala]         [rl_bandit.py]                                                               
 ```                                                                                                                                  
 Python for data, ML, RL; Scala/Spark for streaming analytics.                                                                        
                                                                                                                                      
 ---                                                                                                                                  
                                                                                                                                      
 ## 3. Component Walk-through                                                                                                         
                                                                                                                                      
 | Module / Path                       | Purpose / Main Entry-point                        |                                          
 |------------------------------------- |--------------------------------------------------|                                          
 | `src/read_edf.py`                   | Convert raw EDF to clean FIF (`__main__`)         |                                          
 | `src/preprocess_eeg.py`             | Window/filter/feature-extract EEG (`__main__`)    |                                          
 | `src/simulate_eeg.py`               | Generate synthetic EEG, stream via LSL (`__main__`)|                                         
 | `src/lsl_stream.py`                 | Publish preprocessed windows to LSL (`LSLWindowStreamer`) |                                  
 | `src/dl_models.py`                  | CNN/TCN/Transformer models & loader (`load_pretrained`) |                                    
 | `src/realtime_classifier.py`        | Real-time DL inference, publish state (`__main__`)|                                          
 | `src/rl_bandit.py`                  | LinUCB contextual bandit core (`LinUCBBandit`)    |                                          
 | `src/errp_feedback.py`              | ErrP detection, RL reward loop (`__main__`)       |                                          
 | `scala-streaming/EegStreamProcessor.scala` | Spark Structured Streaming LSL ingestion (`main`) |                                   
                                                                                                                                      
 ---                                                                                                                                  
                                                                                                                                      
 ## 4. Example Workflows                                                                                                              
                                                                                                                                      
 **A. Offline: EDF → FIF → Preprocessing**                                                                                            
 ```bash                                                                                                                              
 python src/read_edf.py data/raw.edf data/clean.fif                                                                                   
 python src/preprocess_eeg.py data/clean.fif --window_sec 7 --overlap_sec 2 --stream_lsl                                              
 ```                                                                                                                                  
                                                                                                                                      
 **B. Full Real-Time Demo (no hardware required)**                                                                                    
 ```bash                                                                                                                              
 # Terminal 1                                                                                                                         
 python src/simulate_eeg.py --duration 30 --sfreq 250 --channels 8 --window 7                                                         
 # Terminal 2                                                                                                                         
 python src/realtime_classifier.py --model cnn --ckpt weights/my_cnn.pt --n_classes 4                                                 
 # Terminal 3                                                                                                                         
 python src/errp_feedback.py                                                                                                          
 ```                                                                                                                                  
                                                                                                                                      
 **C. Spark Streaming Ingestion**                                                                                                     
 ```bash                                                                                                                              
 cd scala-streaming                                                                                                                   
 sbt run                                                                                                                              
 ```                                                                                                                                  
                                                                                                                                      
 ---                                                                                                                                  
                                                                                                                                      
 ## 5. Development & Testing                                                                                                          
                                                                                                                                      
 - **Python:** `black`, `isort`, `ruff`, `mypy`, `pytest`                                                                             
 - **Scala:** `scalafmt`, `sbt test`, `spark-submit` (local)                                                                          
 - **General:** All code in `src/` and `scala-streaming/` is type-checked and linted; see `requirements.txt`.                         
                                                                                                                                      
 ---                                                                                                                                  
                                                                                                                                      
 ## 6. FAQ / Troubleshooting                                                                                                          
                                                                                                                                      
 - **`ModuleNotFoundError: No module named 'pylsl'`**                                                                                 
   → Install pylsl: `pip install pylsl`                                                                                               
 - **`CUDA unavailable` or slow inference**                                                                                           
   → PyTorch will fall back to CPU; check your CUDA install if GPU is required.                                                       
 - **`LSL stream not resolved`**                                                                                                      
   → Ensure the producer (e.g., `simulate_eeg.py`) is running before the consumer; check firewall and network settings.               
                                                                                                                                      
 ---                                                                                                                                  
                                                                                                                                      
 ## 7. License                                                                                                                        
                                                                                                                                      
 Distributed under the MIT License.                                                                                                   
 © 2024 NeuroFeedbackLab
