# -*- coding: utf-8 -*-
"""Reads an EDF file and extracts EEG data."""

import argparse
import mne
import logging
import sys
import numpy as np # For np.union1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Common patterns for reference channel names (case-insensitive)
COMMON_REF_PATTERNS = ['M1', 'M2', 'A1', 'A2', 'TP9', 'TP10', 'REF', 'MASTOID', 'EAR']


def read_edf_eeg(file_path: str):
    """
    Reads an EDF file, identifies EEG channels, and extracts their data.

    Args:
        file_path (str): The path to the EDF file.
                       The function attempts to include common reference channels (e.g., M1, M2, A1, A2, TP9, TP10)
                       if found, alongside channels typed as 'EEG' in the EDF.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The EEG data (channels x samples).
            - list[str]: The names of the extracted EEG channels.
            - float: The sampling frequency in Hz.
            - dict[str, numpy.ndarray] or None: A dictionary mapping EEG channel names to their 3D positions, or None if positions cannot be determined.
        Returns (None, None, None, None) if an error occurs or no EEG channels are found.
    """
    try:
        logging.info(f"Reading EDF file: {file_path}")
        # Load the EDF file, preload data into memory
        # Exclude non-data channels like annotations
        raw = mne.io.read_raw_edf(file_path, preload=True, exclude=['Status', 'Annotations'])

        # --- Enhanced Channel Selection ---
        # 1. Get indices of channels typed as 'EEG'
        eeg_indices = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False, exclude=[])
        
        # 2. Get indices of common reference channels by name pattern
        ref_indices = []
        raw_ch_names_lower = [ch.lower() for ch in raw.ch_names]
        for i, ch_name_lower in enumerate(raw_ch_names_lower):
            if any(pattern.lower() in ch_name_lower for pattern in COMMON_REF_PATTERNS):
                ref_indices.append(i)
        
        # Combine EEG and reference channel indices, ensuring uniqueness
        combined_indices = np.union1d(eeg_indices, ref_indices).astype(int)

        if combined_indices.size == 0:
            logging.warning("No EEG channels (based on type or common reference names) found in the file.")
            return None, None, None, None

        selected_ch_names = [raw.ch_names[i] for i in combined_indices]
        logging.info(f"Channels initially selected (EEG type or common ref name): {selected_ch_names}")

        # Create a new raw object with the selected channels
        raw_selected = raw.copy().pick(picks=selected_ch_names)
        # --- End of Enhanced Channel Selection ---

        # Filter out channels starting with 'cs_' (case-insensitive) as they are often non-data or status channels.
        # This filter is applied AFTER the initial selection of EEG and reference channels.
        initial_ch_names = list(raw_selected.ch_names) # Get a mutable list of current channel names
        channels_to_keep = [ch_name for ch_name in initial_ch_names if not ch_name.lower().startswith('cs_')]

        if not channels_to_keep:
            logging.warning(
                f"All initially identified EEG channels ({initial_ch_names}) were 'cs_' prefixed. "
                f"No primary EEG data channels remain after filtering."
            )
            return None, None, None, None

        # If the list of channels to keep is different from the initial list, then apply the pick.
        if len(channels_to_keep) < len(initial_ch_names):
            channels_dropped = [ch for ch in initial_ch_names if ch not in channels_to_keep]
            logging.info(
                f"Channels after initial EEG/ref selection ({len(initial_ch_names)}): {initial_ch_names}. "
                f"Filtered out 'cs_' prefixed channels: {channels_dropped}. "
                f"Retaining ({len(channels_to_keep)}): {channels_to_keep}."
            )
            raw_selected.pick_channels(channels_to_keep) # Modifies raw_selected in-place
        else:
            logging.info(
                f"All {len(initial_ch_names)} channels from EEG/ref selection retained "
                f"(no 'cs_' prefixed channels found to filter): {initial_ch_names}."
            )
        # Now raw_selected contains only the filtered channels. raw_selected.ch_names is updated.

        if not raw_selected.ch_names: # Double check after cs_ filter
            logging.warning("No channels remained after 'cs_' prefix filtering.")
            return None, None, None, None

        eeg_data = raw_selected.get_data()
        eeg_channels = raw_selected.ch_names
        sfreq = raw_selected.info['sfreq']
        channel_positions = None

        # Attempt to set a standard montage and get channel positions
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw_selected.set_montage(montage, on_missing='warn') # Warn if some channels don't match the montage
            logging.info("Applied standard 10-20 montage.")

            current_montage_obj = raw_selected.get_montage()
            if current_montage_obj:
                # get_positions() returns a dict with 'ch_pos', 'coord_frame', etc.
                # 'ch_pos' is a dict: {ch_name: array([x, y, z])}
                all_montage_positions = current_montage_obj.get_positions()['ch_pos']
                # Filter for the actual EEG channels present in raw_eeg
                channel_positions = {ch_name: all_montage_positions[ch_name]
                                     for ch_name in eeg_channels if ch_name in all_montage_positions}
                if not channel_positions:
                    logging.warning("Could not retrieve positions for any EEG channels after setting montage.")
                    channel_positions = None # Ensure it's None if dict is empty
                else:
                    logging.info(f"Retrieved positions for {len(channel_positions)} out of {len(eeg_channels)} EEG channels.")
            else:
                logging.warning("No montage information could be set or retrieved for EEG channels.")
        except Exception as e_montage:
            logging.warning(f"Could not set or process montage: {e_montage}")
            channel_positions = None

        logging.info(f"Extracted {len(eeg_channels)} EEG channels with sampling frequency {sfreq} Hz.")
        logging.debug(f"EEG Channel names: {eeg_channels}")
        logging.info(f"EEG Data shape: {eeg_data.shape}")

        return eeg_data, eeg_channels, sfreq, channel_positions

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return None, None, None, None
    except Exception as e:
        logging.error(f"An error occurred while processing {file_path}: {e}")
        return None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EEG data from an EDF file.")
    parser.add_argument("edf_file", help="Path to the .edf file")
    parser.add_argument("output_fif_file", help="Path to save the processed EEG data as a .fif file")
    args = parser.parse_args()

    eeg_data, channels, sfreq, positions = read_edf_eeg(args.edf_file)

    if eeg_data is not None and channels is not None and sfreq is not None:
        logging.info("Successfully extracted EEG data. Preparing to save to .fif file.")

        # Create MNE Info object
        # Assuming all returned channels are to be treated as 'eeg' type for saving.
        # This includes actual EEG scalp channels and any reference channels (e.g., M1, M2)
        # that were intentionally kept.
        ch_types = ['eeg'] * len(channels)
        info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
        logging.debug(f"Created MNE Info object for {len(channels)} channels with sfreq {sfreq} Hz.")

        # Create RawArray object
        raw_to_save = mne.io.RawArray(eeg_data, info)

        # Set montage if positions are available
        if positions and isinstance(positions, dict) and positions: # Check if dict is not empty
            # Filter positions to only include channels present in the final `channels` list
            valid_positions = {ch: pos for ch, pos in positions.items() if ch in channels and not np.all(np.isnan(pos))}
            if valid_positions:
                try:
                    montage = mne.channels.make_dig_montage(ch_pos=valid_positions, coord_frame='head')
                    raw_to_save.set_montage(montage)
                    logging.info(f"Applied montage with {len(valid_positions)} channel positions to RawArray.")
                except Exception as e_montage_set:
                    logging.warning(f"Could not create or set montage from positions: {e_montage_set}. Positions: {valid_positions}")
            else:
                logging.info("No valid (non-NaN) channel positions found for the selected channels. Saving without explicit montage.")
        else:
            logging.info("No channel positions available or positions dictionary is empty/all NaN. Saving without explicit montage.")

        # Save to .fif file
        raw_to_save.save(args.output_fif_file, overwrite=True, fmt='single') # Save with float32
        logging.info(f"Processed EEG data saved to: {args.output_fif_file} with single precision.")
    else:
        logging.error("Failed to extract EEG data. Nothing to save.")
        sys.exit(1)
