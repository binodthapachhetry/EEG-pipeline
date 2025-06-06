# -*- coding: utf-8 -*-
"""
Performs windowed preprocessing of EEG data from a .fif file.
Simulates near real-time processing.
"""

import argparse
import logging
import sys
import mne
import numpy as np
# Optional LSL streaming
try:
    from lsl_stream import LSLWindowStreamer
except ImportError:
    LSLWindowStreamer = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define standard EEG frequency bands
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Sigma": (12.0, 15.0), # Often associated with sleep spindles
    "Beta": (15.0, 30.0)   # Ensure h_freq of bandpass is >= 30Hz
}
TOTAL_POWER_BAND = (BANDS["Delta"][0], BANDS["Beta"][1]) # Min of Delta to Max of Beta

def _extract_spectral_features(data_window: np.ndarray, sfreq: float):
    """
    Extracts spectral features (band powers) from a window of EEG data.

    Args:
        data_window (np.ndarray): The EEG data for the window (channels x samples).
        sfreq (float): The sampling frequency.

    Returns:
        dict: A dictionary containing feature names and their values.
              Features are averaged across channels.
    """
    features = {}
    
    # Calculate PSD using Welch's method for all channels
    # n_fft can be sfreq for 1s resolution, or more for finer freq resolution.
    # Ensure fmax covers the highest band of interest.
    psds, freqs = mne.time_frequency.psd_array_welch(
        data_window, sfreq, fmin=TOTAL_POWER_BAND[0], fmax=TOTAL_POWER_BAND[1],
        n_fft=int(sfreq * 2), # 2-second FFT windows for Welch
        n_overlap=int(sfreq * 1), # 1-second overlap
        average='mean', verbose=False
    ) # psds shape: (n_channels, n_freqs)

    # Calculate absolute power for each band
    abs_band_powers = {}
    for band, (fmin, fmax) in BANDS.items():
        band_mask = (freqs >= fmin) & (freqs < fmax)
        # Sum power in band, then average across channels
        abs_band_powers[band] = np.mean(np.sum(psds[:, band_mask], axis=1))
        features[f"abs_{band.lower()}"] = abs_band_powers[band]

    # Calculate total power and relative powers
    total_power = np.sum(list(abs_band_powers.values())) # Sum of powers in defined bands
    if total_power > 0: # Avoid division by zero
        for band, abs_power in abs_band_powers.items():
            features[f"rel_{band.lower()}"] = abs_power / total_power
        features["ratio_alpha_delta"] = abs_band_powers.get("Alpha", 0) / abs_band_powers.get("Delta", 1e-9) # Avoid zero division
        features["ratio_theta_beta"] = abs_band_powers.get("Theta", 0) / abs_band_powers.get("Beta", 1e-9)
    return features

def preprocess_eeg_windowed(
    fif_file_path: str,
    window_duration_sec: float = 7.0,
    overlap_duration_sec: float = 2.0,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: float = 60.0,
    stream_lsl: bool = False
):
    """
    Loads EEG data from a .fif file and processes it in overlapping windows.

    Args:
        fif_file_path (str): Path to the input .fif file.
        window_duration_sec (float): Duration of each processing window in seconds.
        overlap_duration_sec (float): Duration of overlap between consecutive windows in seconds.
        l_freq (float): Low cutoff frequency for bandpass filter.
        h_freq (float): High cutoff frequency for bandpass filter.
        notch_freq (float): Frequency for the notch filter (e.g., 50 or 60 Hz).

    Returns:
        list[tuple[np.ndarray, list[str], dict]]: A list of tuples. Each tuple contains:
            - A NumPy array representing a preprocessed window of EEG data.
            - A list of strings, where each string is a sleep stage annotation (e.g., "NREM2", "REM")
              that overlaps with the window. The list is empty if no annotations overlap.
            - A dictionary of extracted features for the window.
        Returns an empty list if an error occurs or no data is processed.
    """
    try:
        logging.info(f"Loading EEG data from: {fif_file_path}")
        raw = mne.io.read_raw_fif(fif_file_path, preload=True)
        logging.info(f"Successfully loaded data with {len(raw.ch_names)} channels and sampling frequency {raw.info['sfreq']} Hz.")

        # Ensure all channels are treated as EEG for this processing script
        # This is a general assumption; specific channel type handling might be needed for advanced steps
        raw.set_channel_types({ch_name: 'eeg' for ch_name in raw.ch_names})

    except FileNotFoundError:
        logging.error(f"Error: File not found at {fif_file_path}")
        return []
    except Exception as e:
        logging.error(f"Error loading {fif_file_path}: {e}")
        return []

    sfreq = raw.info['sfreq']
    step_duration_sec = window_duration_sec - overlap_duration_sec
    if step_duration_sec <= 0:
        logging.error("Overlap duration must be less than window duration.")
        return []

    processed_windows_data = []
    lsl_streamer = None
    if stream_lsl and LSLWindowStreamer:
        lsl_streamer = LSLWindowStreamer(raw.ch_names, sfreq, int(window_duration_sec*sfreq))
    num_windows = 0

    # Calculate window start times
    # Start of the first window is 0.
    # Start of subsequent windows is `step_duration_sec` after the previous.
    window_starts_sec = np.arange(0, raw.times[-1] - window_duration_sec + 1/sfreq, step_duration_sec)

    for i, t_start_sec in enumerate(window_starts_sec):
        t_end_sec = t_start_sec + window_duration_sec

        # Create a view for the current window
        # Crop creates a copy, ensuring original raw is not modified
        raw_window = raw.copy().crop(tmin=t_start_sec, tmax=t_end_sec, include_tmax=False) # tmax is exclusive

        # Determine sleep stage(s) for the current window
        current_window_stages = []
        if hasattr(raw, 'annotations') and raw.annotations is not None and len(raw.annotations) > 0:
            for ann in raw.annotations:
                ann_start = ann['onset']
                ann_end = ann['onset'] + ann['duration']
                # Check for overlap: annotation intersects with [t_start_sec, t_end_sec)
                if ann_start < t_end_sec and ann_end > t_start_sec:
                    current_window_stages.append(ann['description'])
            current_window_stages = sorted(list(set(current_window_stages))) # Unique stages, sorted

        stage_info_log = f"Stage(s): {', '.join(current_window_stages) if current_window_stages else 'N/A'}"
        logging.info(f"Processing window {i+1}: {t_start_sec:.2f}s - {t_end_sec:.2f}s. {stage_info_log}")

        # 1. Re-referencing (Average Reference)
        # projection=False applies the reference directly.
        # For average reference, it's good practice to ensure only EEG channels are used.
        # Here, we've set all to EEG, which is common for average ref.
        try:
            raw_window.set_eeg_reference('average', projection=False, verbose=False)
        except Exception as e_ref:
            logging.warning(f"Could not apply average reference to window {i+1}: {e_ref}. Skipping re-referencing for this window.")

        # 2. Band-pass filtering
        raw_window.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)

        # 3. Notch filtering
        if notch_freq is not None and notch_freq > 0:
            raw_window.notch_filter(freqs=notch_freq, fir_design='firwin', verbose=False)

        # 4. Feature Extraction
        window_data_array = raw_window.get_data()
        extracted_features = _extract_spectral_features(window_data_array, sfreq)
        processed_windows_data.append((window_data_array, current_window_stages, extracted_features))
        # --- Real-time push (optional) ---
        if lsl_streamer:
            lsl_streamer.push_window(window_data_array, extracted_features, current_window_stages)
        num_windows += 1

    logging.info(f"Finished processing. Total windows processed: {num_windows}")
    return processed_windows_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EEG data from a .fif file in overlapping windows.")
    parser.add_argument("input_fif_file", help="Path to the input .fif file (output from read_edf.py).")
    parser.add_argument("--window_sec", type=float, default=7.0, help="Duration of the processing window in seconds.")
    parser.add_argument("--overlap_sec", type=float, default=2.0, help="Overlap between windows in seconds.")
    parser.add_argument("--l_freq", type=float, default=0.5, help="Low cutoff frequency for bandpass filter (Hz).")
    parser.add_argument("--h_freq", type=float, default=40.0, help="High cutoff frequency for bandpass filter (Hz).")
    parser.add_argument("--notch_freq", type=float, default=60.0, help="Notch filter frequency (Hz). Set to 0 to disable.")
    parser.add_argument("--stream_lsl", action="store_true", help="If set, stream each processed window via LSL.")
    args = parser.parse_args()

    processed_data_list = preprocess_eeg_windowed(
        args.input_fif_file, args.window_sec, args.overlap_sec,
        args.l_freq, args.h_freq, args.notch_freq, args.stream_lsl
    )

    if processed_data_list:
        logging.info(f"Successfully processed {len(processed_data_list)} windows.")
        # Example: print shape and stages of the first processed window
        # first_window_data, first_window_stages, first_window_features = processed_data_list[0]
        # logging.info(f"Shape of first processed window: {first_window_data.shape}")
        # logging.info(f"Sleep stage(s) for first window: {', '.join(first_window_stages) if first_window_stages else 'N/A'}")
        # logging.info(f"Features for first window: {first_window_features}")
    else:
        logging.warning("No data was processed.")
        sys.exit(1)
