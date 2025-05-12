# -*- coding: utf-8 -*-
"""Reads an EDF file and extracts EEG data."""

import argparse
import mne
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_edf_eeg(file_path: str):
    """
    Reads an EDF file, identifies EEG channels, and extracts their data.

    Args:
        file_path (str): The path to the EDF file.

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

        # Pick only EEG channels. MNE attempts automatic detection based on names.
        # If specific channel names are known, use raw.pick_channels(['EEG Fpz-Cz', ...])
        raw_eeg = raw.copy().pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)

        if not raw_eeg.ch_names:
            logging.warning("No EEG channels found in the file.")
            return None, None, None, None

        eeg_data = raw_eeg.get_data()
        eeg_channels = raw_eeg.ch_names
        sfreq = raw_eeg.info['sfreq']
        channel_positions = None

        # Attempt to set a standard montage and get channel positions
        try:
            # Use a standard 10-20 montage
            montage = mne.channels.make_standard_montage('standard_1020')
            raw_eeg.set_montage(montage, on_missing='warn') # Warn if some EEG channels don't match the montage
            logging.info("Applied standard 10-20 montage.")

            # Extract positions for the EEG channels present in raw_eeg
            current_montage_obj = raw_eeg.get_montage()
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
    args = parser.parse_args()

    eeg_data, channels, sfreq, positions = read_edf_eeg(args.edf_file)

    if eeg_data is not None:
        logging.info("Successfully extracted EEG data.")
        # Example: Print shape or save data if needed
        # print(f"Data shape: {eeg_data.shape}")
        # print(f"Channels: {channels}")
        # print(f"Sampling Frequency: {sfreq} Hz")
        if positions:
            logging.info(f"Channel positions: {positions}")
        # else:
        #     logging.info("Channel positions could not be determined.")
    else:
        logging.error("Failed to extract EEG data.")
        sys.exit(1)
