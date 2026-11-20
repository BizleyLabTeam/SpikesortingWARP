### Note: Rec A / Rec B may correspond to Left or Right depending on experiment setup.
### Specify 'rec_side' ('A' or 'B') in the configuration to avoid ambiguity.


# This code adjustments are based on Python 3.11.4 and SpikeInterface version 0.101.0 or later.
# First download spikesorting_scripts from Jules's github

# Set logging before the rest as neo (and neo-based imports) needs to be imported after logging has been set
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('multirec_sorting')
logger.setLevel(logging.DEBUG)

# Required imports are listed before each function but already included in the initial import section.
import argparse
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import time
import csv
from pathlib import Path
from sklearn.decomposition import PCA

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jsmin import jsmin
from tqdm import tqdm

import spikeinterface as si
import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.exporters as sexp
import spikeinterface.postprocessing as spost
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
from spikeinterface.extractors import NumpyRecording, read_mcsh5
from spikeinterface.exporters import export_to_phy
from spikeinterface import extract_waveforms
from spikeinterface.core import create_sorting_analyzer
from spikeinterface.preprocessing import remove_artifacts
from spikeinterface.core import get_random_data_chunks

from spikeinterface import append_recordings, concatenate_recordings
from probeinterface import generate_multi_columns_probe

from spikesorting_scripts.helpers import generate_warp_32ch_probe
from spikesorting_scripts.preprocessing import (
    build_qc_summary_table,
    evaluate_block_qc,
    evaluate_orientation_qc,
    format_qc_summary_markdown,
    load_warp_channel_map,
    plot_channel_coherence,
    plot_orientation_qc,
    remove_disconnection_events,
    sample_block_qc_metrics,
    write_block_qc_csv,
    write_orientation_qc_csv,
    write_qc_summary_tables,
)


#  <compute_rec_power> function computes the average power of a recording
def compute_rec_power(rec):
    subset_data = sc.get_random_data_chunks(rec, num_chunks_per_segment=100,
                                            ## You should change it according to size of your input frame! number of random chunks sampled from each segment of the data
                                            chunk_size=10000,
                                            ## This defines the size (number of frames or samples) of each chunk.
                                            ## Equation for adjusting: chunk_size*num_chunks_per_segment <= total size of data
                                            seed=0,  ## Change it if need different random chunk for testing
                                            )
    power = np.mean(np.abs(subset_data))
    return power


# <compute_channel_powers> Divide the recording into 100 chunks, capturing approximately 4% of a 20-minute session for fs = 20K

def compute_channel_powers(recording, num_chunks=100, chunk_size=10000, seed=0):
    """
    Computes the absolute power of each channel by averaging absolute values over random chunks.
    """
    subset_data = get_random_data_chunks(recording, num_chunks_per_segment=num_chunks,
                                         chunk_size=chunk_size, seed=seed)
    channel_powers = np.mean(np.abs(subset_data), axis=0)  # Mean absolute value per channel
    return channel_powers


# <remove_static_flat_periods> Removes continuous absolute flat periods in the recording and replaces them with the median of surrounding data

from spikeinterface.core import NumpyRecording, BinaryRecordingExtractor

logger = logging.getLogger(__name__)


def remove_static_flat_periods(
    recording, flat_threshold=1e-9, min_flat_duration_s=0.4,
    padding_samples=20, probe=None, output_dir="preprocessed_recording",
    channel_permutation=None,
):
    if recording is None:
        raise ValueError("Input recording is None")

    if recording.get_num_segments() != 1:
        raise ValueError("Flat-period cleaning expects one session with one segment")

    sampling_frequency = recording.get_sampling_frequency()
    min_flat_samples = int(min_flat_duration_s * sampling_frequency)

    traces = recording.get_traces()
    if channel_permutation is not None:
        traces = traces[:, channel_permutation]
    new_traces = traces.astype(np.float32)
    total_flat_segments = 0

    for ch in range(traces.shape[1]):
        channel_data = traces[:, ch]
        flat_mask = np.abs(np.diff(channel_data)) < flat_threshold
        flat_starts = np.where(np.diff(np.concatenate(([False], flat_mask, [False]))))[0][::2]
        flat_ends = np.where(np.diff(np.concatenate(([False], flat_mask, [False]))))[0][1::2]

        for start, end in zip(flat_starts, flat_ends):
            if (end - start) >= min_flat_samples:
                total_flat_segments += 1
                start = max(0, start - padding_samples)
                end = min(len(channel_data), end + padding_samples)

                if start >= 100:
                    median_window = channel_data[start - 100:start]
                elif end + 100 < len(channel_data):
                    median_window = channel_data[end:end + 100]
                else:
                    median_window = channel_data

                replacement_value = np.median(median_window) if len(median_window) > 0 else np.median(channel_data)
                new_traces[start:end, ch] = replacement_value

    print(f"Total flat segments detected: {total_flat_segments}")

    original_t_starts = getattr(recording, "_t_starts", None)
    t_starts = original_t_starts if original_t_starts else [0]

    numpy_recording = NumpyRecording(
        traces_list=[new_traces],
        channel_ids=recording.get_channel_ids(),
        sampling_frequency=sampling_frequency,
        t_starts=t_starts
    )

    # Copy channel locations if present
    channel_ids = recording.get_channel_ids()
    if "location" in recording.get_property_keys():
        numpy_recording.set_property("location", recording.get_property("location"))

    # Write to binary format
    os.makedirs(output_dir, exist_ok=True)
    binary_file_path = os.path.join(output_dir, "preprocessed_data.dat")

    BinaryRecordingExtractor.write_recording(
        numpy_recording,
        file_paths=binary_file_path,
        dtype="float32",
        n_jobs=1,
        chunk_duration="1s",
        verbose=True
    )

    binary_recording = BinaryRecordingExtractor(
        file_paths=binary_file_path,
        sampling_frequency=sampling_frequency,
        num_channels=new_traces.shape[1],
        channel_ids=channel_ids,
        t_starts=t_starts,
        dtype="float32"
    )

    # Copy channel properties again to binary recording
    if "location" in numpy_recording.get_property_keys():
        binary_recording.set_property("location", numpy_recording.get_property("location"))

    # Apply probe if available
    try:
        if probe is not None:
            binary_recording.set_probe(probe)
        else:
            rec_probe = recording.get_probe()
            if rec_probe is not None:
                binary_recording.set_probe(rec_probe)
            else:
                logger.warning("No probe found or provided for binary_recording.")
    except Exception as e:
        logger.warning(f"Failed to set probe on binary_recording: {e}")

    # Final info
    print(f"Original Sampling Frequency: {sampling_frequency}")
    print(f"Binary Num Segments: {binary_recording.get_num_segments()}")
    print(f"Session cleaned and saved to: {binary_file_path}")

    return binary_recording


# <PCA> By default, the input consists of 2 components, reduced by a 0.5 shared component. **** Convert Kath?s PCA MATLAB code into Python ****

from spikeinterface.core import NumpyRecording

logger = logging.getLogger(__name__)


def remove_shared_noise_pca(recording, chunk_size, n_components=2, strength=0.5):
    """
    Removes shared noise across channels using PCA-based subtraction.

    Parameters:
    recording : Recording
    chunk_size : Number of samples per chunk for PCA cleaning.
    n_components : Number of PCA components to remove.
    strength : Proportion of shared component to subtract (0 = none, 1 = full subtraction).

    Returns
    NumpyRecording
        Cleaned recording with shared noise reduced.
    """
    sampling_rate = recording.get_sampling_frequency()
    channel_ids = recording.get_channel_ids()
    total_samples = recording.get_num_frames()

    cleaned_chunks = []

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = recording.get_traces(start_frame=start, end_frame=end)

        if chunk.shape[0] < 2 or chunk.shape[1] <= n_components:
            logger.warning(f"Skipping PCA chunk {start}-{end}: insufficient data or channels.")
            cleaned_chunks.append(chunk)
            continue

        chunk_centered = chunk - np.mean(chunk, axis=0)

        if not np.isfinite(chunk_centered).all():
            logger.warning(f"Skipping PCA chunk {start}-{end}: contains NaNs/Infs.")
            cleaned_chunks.append(chunk)
            continue

        if np.all(np.std(chunk_centered, axis=0) == 0):
            logger.warning(f"Skipping PCA chunk {start}-{end}: zero variance across all channels.")
            cleaned_chunks.append(chunk)
            continue

        try:
            pca = PCA(n_components=n_components)
            projected = pca.fit_transform(chunk_centered)
            reconstructed = pca.inverse_transform(projected)
            cleaned_chunk = chunk_centered - strength * reconstructed
            cleaned_chunks.append(cleaned_chunk)
        except Exception as e:
            logger.warning(f"PCA failed on chunk {start}-{end}: {e}")
            cleaned_chunks.append(chunk)  # fallback to original

    if not cleaned_chunks:
        raise ValueError("All PCA chunks were skipped-output would be empty.")

    # Combine all cleaned chunks
    cleaned_data = np.vstack(cleaned_chunks)

    # Wrap cleaned data in a NumpyRecording
    cleaned_recording = NumpyRecording(
        traces_list=[cleaned_data],
        sampling_frequency=sampling_rate,
        channel_ids=channel_ids
    )

    # Try to preserve the probe if possible
    try:
        probe = recording.get_probe()
        if probe is not None:
            cleaned_recording.set_probe(probe)
        else:
            logger.warning("remove_shared_noise_pca: Input recording has no probe.")
    except Exception as e:
        logger.warning(f"Could not copy probe: {e}")

    # Copy channel level properties like 'location', 'group'
    for prop in recording.get_property_keys():
        try:
            cleaned_recording.set_property(prop, recording.get_property(prop))
        except Exception as e:
            logger.warning(f"Failed to set property '{prop}': {e}")

    return cleaned_recording


#  <log_channel_removal> removed bad channels to 'channel_removed.csv' in the structured label folder.

def log_channel_removal(bad_channel_names, output_folder, ferret_name, Brain_part, Group_name):
    label_folder = f"{ferret_name}_{Brain_part}__{Group_name}"
    output_dir = Path(output_folder) / label_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "channel_removed.csv"
    header = ['bad_channels']
    row = [','.join(map(str, bad_channel_names))]

    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


#  <log_recording_table>  logs recording metadata to a recording_info.csv in a structured subfolder
def log_recording_table(block_name, trace_shape, output_folder, ferret_name, Brain_part, Group_name):
    # Build label-specific output directory
    label_folder = f"{ferret_name}_{Brain_part}__{Group_name}"
    output_dir = Path(output_folder) / label_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define CSV path
    csv_filename = "recording_info.csv"
    output_path = output_dir / csv_filename

    # Create or append to the CSV
    header = ['block', 'trace_shape']
    row = [block_name, str(trace_shape)]

    write_header = not output_path.exists()
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    return output_path


#  <log_bad_block>  removed bad blocks to 'bad_blocks.csv' in the structured label folder.
def log_bad_block(block_name, power_value, output_folder, ferret_name, Brain_part, Group_name):
    label_folder = f"{ferret_name}_{Brain_part}__{Group_name}"
    output_dir = Path(output_folder) / label_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = "bad_blocks.csv"
    output_path = output_dir / csv_filename

    header = ['block', 'power']
    row = [block_name, power_value]

    write_header = not output_path.exists()
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    return output_path


# <preprocess_rec> function sets the probe (adjust the probe function according to your data type) and applies filtering.

def preprocess_rec(recording, output_dir="plots", output_folder=None, ferret_name=None, Brain_part=None,
                   Group_name=None):
    # Step 1:Sets probe
    logger.info('set_probe')
    probe = generate_warp_32ch_probe()
    recording = recording.set_probe(probe)
    print(f"Probe assigned: {recording.get_channel_ids()}")
    print(probe)

    # Step 2: Compute absolute power for each channel and remove the bad channel
    logger.info('remove the bad channel')
    channel_powers = compute_channel_powers(recording)
    mean_power = np.median(channel_powers)
    high_power_threshold = 3 * mean_power  # Threshold for bad channels
    bad_channels = np.where(channel_powers > high_power_threshold)[0]

    if len(bad_channels) > 0:
        print(f"Removing {len(bad_channels)} channels with excessive power: {bad_channels}")
        # recording = recording.remove_channels(bad_channels)
        bad_channel_names = [recording.get_channel_ids()[i] for i in bad_channels]
        recording = recording.remove_channels(bad_channel_names)
        # save those names
        log_channel_removal(bad_channel_names, output_folder=output_folder, ferret_name=ferret_name,
                            Brain_part=Brain_part, Group_name=Group_name)
        print(f"Channels after removal: {recording.get_channel_ids()}")
        print(f"Remaining channels after removal: {recording.get_num_channels()}")

    # Step 3: Flat periods were already processed in each retained session.
    logger.info('Flat periods already processed per session before concatenation')
    recording_pre = recording

    # Step 4: common_reference
    logger.info('common_reference')
    recording_cmr = spre.common_reference(recording_pre, reference='global', operator='median')
    logger.info('remove_disconnection_events')

    # Step 5: remove artifacts caused by transients
    logger.info('remove_disconnection_events')
    recording_de = remove_disconnection_events(recording_cmr,
                                               compute_medians="random",
                                               chunk_size=int(recording_cmr.get_sampling_frequency() * 3),
                                               n_median_threshold=3,
                                               n_peaks=0,
                                               )

    # Step 6: bandpass_filter
    logger.info('bandpass_filter')
    recording_bf = spre.bandpass_filter(recording_de, freq_min=300,
                                        freq_max=6000)  ##Adjust this according to your requirements. The maximum frequency should be less than fs/2

    logger.info('remove_shared_noise_pca')
    chunk_size = int(recording_bf.get_sampling_frequency())  # 1 second chunk
    recording_pca = remove_shared_noise_pca(recording_bf, chunk_size=chunk_size, n_components=2, strength=0.5)

    # Step 7: whitenning
    logger.info('whitenning')
    recording_wf = spre.whiten(recording_bf, dtype='float32')

    return recording_wf

# <export_all> function extracts waveforms and saves all necessary requirements for use in GUI-Phy.

def export_all(sortername, sortextract, recextract, rec_name, ferret_name, Brain_part, Group_name, working_directory,
               output_folder, job_kwargs):
    output_folder = Path(output_folder)
    if rec_name == "data" and output_folder.parts[-1] == "data":
        rec_name = ""

    label_folder = f"{ferret_name}_{Brain_part}__{Group_name}"
    outDir = output_folder / label_folder / rec_name / sortername

    #    outDir = output_folder / rec_name / sortername

    # Handle pre-existing folder manually
    waveform_folder = outDir / 'waveforms'

    if waveform_folder.exists():
        shutil.rmtree(waveform_folder)  # Delete existing folder to avoid conflicts

    logger.info("Creating SortingAnalyzer...")
    we = sc.extract_waveforms(recextract,
                              sortextract, waveform_folder,  # outDir / 'waveforms'
                              ms_before=2.5, ms_after=3,
                              max_spikes_per_unit=500,  # load_if_exists=True, 500
                              overwrite=None,
                              **job_kwargs
                              # n_jobs=10,
                              # chunk_size=30000
                              )

    try:
        sorting_analyzer = sc.create_sorting_analyzer(
            sortextract,
            we.recording,  # recextract,
            overwrite=None,
            **job_kwargs
        )
        logger.info(f"Type of 'sorting_analyzer': {type(sorting_analyzer)}")
    except Exception as e:
        logger.error(f"Error creating SortingAnalyzer: {e}")
        raise

    logger.info(f"Type of 'sorting_analyzer': {type(sorting_analyzer)}")
    logger.info(f"SortingAnalyzer attributes: {dir(sorting_analyzer)}")

    # Selects a random subset of spikes for each unit, which is a prerequisite for subsequent computations like waveforms and templates
    sorting_analyzer.compute('random_spikes')
    # Explicitly compute 'waveforms' extension
    sorting_analyzer.compute('waveforms')
    # Now compute 'templates' extension
    sorting_analyzer.compute('templates')

    # Export to Phy using SortingAnalyzer
    logger.info("Exporting to Phy format...")
    try:
        sexp.export_to_phy(
            sorting_analyzer,
            outDir / 'phy',
            remove_if_exists=True,
            copy_binary=True,
            **job_kwargs
        )
        logger.info(f"Exported Phy files to {outDir / 'phy'}")
    except Exception as e:
        logger.error(f"Error during export_to_phy: {e}")
        raise

    logger.info(f"Processing complete. Phy files saved at: {outDir / 'phy'}")


# =====================================================================
# FLEXIBLE INPUT & RECORDING SIDE MATCHING HELPERS
# =====================================================================

REC_SIDE_PATTERNS = {
    "A": re.compile(r"(?i)(?:^|[^a-zA-Z])rec[\s_-]*a(?:[\s_-]*\d+)?(?:$|[^a-zA-Z0-9])"),
    "B": re.compile(r"(?i)(?:^|[^a-zA-Z])rec[\s_-]*b(?:[\s_-]*\d+)?(?:$|[^a-zA-Z0-9])"),
}


def normalize_rec_side(value):
    """
    Normalizes a recording side specification into 'A', 'B', None, or a custom regex string.
    Accepts values such as:
      - 'A', 'a', 'RecA', 'Rec_A', 'Rec A', 'Rec-A', 'Rec_A_2', 'Left_RecA' -> 'A'
      - 'B', 'b', 'RecB', 'Rec_B', 'Rec B', 'Rec-B', 'Rec_B_2', 'Right_RecB' -> 'B'
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    if text.upper() in ("A", "B"):
        return text.upper()

    match_a = REC_SIDE_PATTERNS["A"].search(text)
    match_b = REC_SIDE_PATTERNS["B"].search(text)
    if match_a and not match_b:
        return "A"
    if match_b and not match_a:
        return "B"

    return text


def extract_block_number(block_name):
    """Extracts the block number from names like 'Block_J1-1128' -> 1128."""
    match = re.search(r"(\d+)$", block_name)
    if match:
        return int(match.group(1))
    digits = re.sub(r"\D", "", block_name)
    return int(digits) if digits else 0


def resolve_datadir(params):
    """
    Resolves the directory containing Block_J* folders.
    Supports both:
      1. datadir / rec_name (e.g. datadir='Z:/WARPData', rec_name='F2206_Hurricane')
      2. datadir directly containing Block_J* folders (e.g. datadir='Z:/WARPData/F2206_Hurricane')
    """
    base_dir = Path(params["datadir"])
    rec_name = params.get("rec_name", "")
    candidate = base_dir / rec_name if rec_name else base_dir

    if candidate.exists() and any(candidate.glob("Block_J*")):
        return candidate
    if base_dir.exists() and any(base_dir.glob("Block_J*")):
        return base_dir
    return candidate


def select_block_h5_file(block_dir: Path, rec_side=None, fallback_side=None):
    """
    Selects the single target .h5 file from a block directory.
    Works for:
      - Pre-split directories (1 .h5 file per block folder)
      - Combined directories (both Rec A and Rec B .h5 files in the same block folder,
        with flexible names like Rec_A.h5, Rec_A_3.h5, Rec_B_2.h5, RecB.h5, etc.)
    """
    h5_files = sorted(block_dir.rglob("*.h5"))
    if not h5_files:
        logger.error(f"No .h5 files found in {block_dir}")
        return None

    effective_side = rec_side or fallback_side

    if effective_side in ("A", "B"):
        target_pattern = REC_SIDE_PATTERNS[effective_side]
        other_side = "B" if effective_side == "A" else "A"
        other_pattern = REC_SIDE_PATTERNS[other_side]

        matched = [f for f in h5_files if target_pattern.search(f.stem)]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            logger.error(
                f"Multiple .h5 files matched Rec {effective_side} in {block_dir}: {[f.name for f in matched]}"
            )
            return None

        # If no explicit Rec A/B pattern matched in the filename, allow a single generic .h5
        # file (e.g. 'recording.h5' in a pre-split folder) as long as it isn't tagged as the opposite side.
        if len(h5_files) == 1 and not other_pattern.search(h5_files[0].stem):
            return h5_files[0]

        logger.error(
            f"Expected 1 .h5 matching Rec {effective_side} in {block_dir}, "
            f"found 0 among: {[f.name for f in h5_files]}"
        )
        return None

    if effective_side is not None:
        custom_re = re.compile(effective_side, re.IGNORECASE)
        matched = [f for f in h5_files if custom_re.search(f.stem) or custom_re.search(f.name)]
        if len(matched) == 1:
            return matched[0]
        logger.error(
            f"Expected 1 .h5 matching '{effective_side}' in {block_dir}, "
            f"found {len(matched)} among: {[f.name for f in h5_files]}"
        )
        return None

    # No side specified or inferred: works when each block folder has only 1 .h5 file
    if len(h5_files) == 1:
        return h5_files[0]

    logger.error(
        f"Expected 1 .h5 in {block_dir}, found {len(h5_files)}: {[f.name for f in h5_files]}. "
        f"Please set '\"rec_side\": \"A\"' or '\"rec_side\": \"B\"' in your JSON config to sort one side."
    )
    return None


############################### MAIN #######################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="path to the json file containing the parameters")
    parser.add_argument(
        "--rec-side",
        default=None,
        help="Optional override for recording side to sort ('A' or 'B', e.g. Rec_A vs Rec_B)",
    )
    parser.add_argument(
        "--block-qc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable enhanced wireless dropout & noise block QC (overrides JSON block_qc.enabled)",
    )
    parser.add_argument(
        "--orientation-qc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable headstage orientation QC across blocks (overrides JSON orientation_qc.enabled)",
    )
    parser.add_argument(
        "--orientation-action",
        choices=["warn", "flip", "exclude"],
        default=None,
        help="Action when upside-down headstage is detected ('warn', 'flip', or 'exclude')",
    )
    parser.add_argument(
        "--reference-block",
        type=str,
        default=None,
        help="Reference block name or number for orientation QC (e.g. '906' or 'Block_J1-906')",
    )
    parser.add_argument(
        "--qc-only",
        action="store_true",
        help="Run block & orientation QC, export summary tables/plots, and exit before concatenation and sorting",
    )
    args = parser.parse_args()

    with open(args.params_file) as json_file:
        minified = jsmin(json_file.read())
        params = json.loads(minified)

    block_qc_cfg = dict(params.get("block_qc") or {})
    if args.block_qc is not None:
        block_qc_cfg["enabled"] = bool(args.block_qc)
    elif args.qc_only:
        block_qc_cfg["enabled"] = True

    orientation_qc_cfg = dict(params.get("orientation_qc") or {})
    if args.orientation_qc is not None:
        orientation_qc_cfg["enabled"] = bool(args.orientation_qc)
    elif args.qc_only:
        orientation_qc_cfg["enabled"] = True
    if args.orientation_action is not None:
        orientation_qc_cfg["action"] = args.orientation_action
    if "reference_block" in params and "reference_block" not in orientation_qc_cfg:
        orientation_qc_cfg["reference_block"] = params["reference_block"]
    if args.reference_block is not None:
        orientation_qc_cfg["reference_block"] = args.reference_block

    block_qc_enabled = bool(block_qc_cfg.get("enabled", False))
    orientation_qc_enabled = bool(orientation_qc_cfg.get("enabled", False))

    chan_map_info = None
    if orientation_qc_enabled:
        raw_map_path = orientation_qc_cfg.get(
            "chan_map_path", "docs/Warp_to_WirelessHeadstage_ChanMap.txt"
        )
        map_path = Path(raw_map_path)
        if not map_path.is_absolute():
            repo_candidate = Path(__file__).resolve().parent / map_path
            if repo_candidate.exists():
                map_path = repo_candidate
        chan_map_info = load_warp_channel_map(map_path)
        logger.info(f"Loaded WARP channel map for orientation QC from {map_path}")
    # Preserve current-directory resolution for configured relative paths.
    config_dir = Path.cwd()
    raw_output = Path(params['output_folder'])
    output_folder = (config_dir / raw_output).resolve() if not raw_output.is_absolute() else raw_output.resolve()
    raw_log = Path(params['logpath'])
    logpath = (config_dir / raw_log).resolve() if not raw_log.is_absolute() else raw_log.resolve()
    logpath.mkdir(parents=True, exist_ok=True)
    # Retrieve the working directory from params or use a default
    now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    fh = logging.FileHandler(logpath / f'multirec_warp_sorting_logs_{now}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info('Starting')
    sorter_list = params['sorter_list']
    logger.info(f'sorter list: {sorter_list}')

    if 'kilosort2' in sorter_list:
        ss.Kilosort2Sorter.set_kilosort2_path(params['sorter_paths']['kilosort2_path'])
    if 'waveclus' in sorter_list:
        ss.WaveClusSorter.set_waveclus_path(params['sorter_paths']['waveclus_path'])
    if 'kilosort3' in sorter_list:
        ss.Kilosort3Sorter.set_kilosort3_path(params['sorter_paths']['kilosort3_path'])

    datadir = resolve_datadir(params)
    print(f"Data directory: {datadir}")
    streams = params[
        'streams']  ## 'streams' is the path to the stream related to your data recording, containing values (channels * frames) in .h5 groups

    blocks = [bl.name for bl in datadir.glob('Block_J*') if bl.is_dir()]
    blocks.sort(key=extract_block_number)

    # Optional block_range filter: e.g. "block_range": [1100, 1130] (inclusive)
    block_range = params.get("block_range")
    if block_range and len(block_range) == 2:
        start_b, end_b = int(block_range[0]), int(block_range[1])
        blocks = [b for b in blocks if start_b <= extract_block_number(b) <= end_b]
        logger.info(f"Filtered blocks to range [{start_b}, {end_b}]: {len(blocks)} blocks")

    # Determine which side (Rec A or Rec B) to sort, if specified or inferable
    rec_side = normalize_rec_side(args.rec_side if args.rec_side is not None else params.get("rec_side"))
    inferred_side = normalize_rec_side(params.get("rec_name"))
    if inferred_side not in ("A", "B"):
        inferred_side = normalize_rec_side(params.get("Brain_part"))
    fallback_side = inferred_side if inferred_side in ("A", "B") else None

    # If side is not specified and cannot be inferred from names, check if blocks contain
    # multiple .h5 files (e.g. combined Rec A and Rec B). Force explicit specification
    # because Rec A / Rec B do not always map to Left / Right.
    if rec_side is None and fallback_side is None and blocks:
        combined_sample = None
        for b in blocks[:5]:
            sample_h5s = sorted((datadir / b).rglob("*.h5"))
            if len(sample_h5s) > 1:
                combined_sample = (b, sample_h5s)
                break

        if combined_sample is not None:
            b_name, sample_h5s = combined_sample
            prompt_side = None
            if sys.stdin.isatty():
                try:
                    available = [s for s in ("A", "B") if any(REC_SIDE_PATTERNS[s].search(f.stem) for f in sample_h5s)]
                    options_str = "/".join(available) if available else "A/B"
                    print(f"\n[Recording Side Required] Multiple recording files found in {b_name}: {[f.name for f in sample_h5s]}")
                    print("Note: Rec A and Rec B do not always correspond to Left and Right.")
                    ans = input(f"Which recording side do you want to sort? [{options_str}]: ").strip().upper()
                    if ans in ("A", "B"):
                        prompt_side = ans
                except (EOFError, KeyboardInterrupt):
                    prompt_side = None

            if prompt_side:
                rec_side = prompt_side
                logger.info(f"Selected recording side via interactive prompt: Rec {rec_side}")
                print(f"Selected recording side: Rec {rec_side}")
            else:
                raise ValueError(
                    f"Multiple recording files found in '{b_name}' ({[f.name for f in sample_h5s]}). "
                    "Because Rec A and Rec B do not always correspond to Left and Right, you must explicitly specify "
                    "which recording side to sort.\n"
                    "Please set '\"rec_side\": \"A\"' or '\"rec_side\": \"B\"' in your JSON config, include 'RecA'/'RecB' "
                    "in 'rec_name' or 'Brain_part', or pass '--rec-side A' / '--rec-side B' on the command line."
                )

    if rec_side:
        logger.info(f"Target recording side configured: {rec_side}")
        print(f"Target recording side: Rec {rec_side}" if rec_side in ("A", "B") else f"Target pattern: {rec_side}")
    elif fallback_side:
        logger.info(f"Inferred recording side from config labels: Rec {fallback_side}")
        print(f"Inferred recording side from config labels: Rec {fallback_side}")

    # Build a side-aware run name for working/recording directories so sorting A and B
    # from the same parent folder never overwrites each other's intermediate files.
    effective_side_tag = rec_side if rec_side in ("A", "B") else fallback_side
    run_rec_name = params["rec_name"]
    if effective_side_tag and not REC_SIDE_PATTERNS[effective_side_tag].search(run_rec_name):
        run_rec_name = f"{run_rec_name}_Rec{effective_side_tag}"

    raw_work = Path(params['working_directory'])
    base_work = (config_dir / raw_work).resolve() if not raw_work.is_absolute() else raw_work.resolve()
    working_directory = base_work / run_rec_name
    working_directory.mkdir(parents=True, exist_ok=True)

    print(blocks)
    pbar = tqdm(blocks)

    recording_list = {stream: [] for stream in streams}

    for stream in streams:
        powers = []
        block_metrics = []
        rec_list = []
        logger.info(f'Loading stream {stream}')
        block_names = {stream: [] for stream in streams}

        for block in pbar:
            pbar.set_postfix_str(f'loading {block}')
            logger.info(f'Loading block {block}')
            print(f"Loading block {block}")

            try:
                h5_file = select_block_h5_file(
                    datadir / block,
                    rec_side=rec_side,
                    fallback_side=fallback_side,
                )
                if h5_file is None:
                    continue
                logger.info(f"Selected .h5 file for {block}: {h5_file.name}")
                print(f"  Using file: {h5_file.name}")

                # --- load recording ---
                rec = se.read_mcsh5(str(h5_file), stream_id=stream)

                if rec is not None and rec.get_num_channels() != 32:
                    logger.warning(
                        f"Stream {stream} in {h5_file.name} has {rec.get_num_channels()} channels "
                        f"(channels={rec.get_channel_ids()}). The 32-channel WARP probe expects 32 channels. "
                        f"Note: In MCS W2100 HDF5 files, stream 0 is often 'Quality Data' (1 channel) and "
                        f"stream 1 is 'Electrode Raw Data' (32 channels)."
                    )

                num_frame_save = rec.get_num_frames()

                log_recording_table(
                    block_name=block,
                    trace_shape=num_frame_save,
                    output_folder=output_folder,
                    ferret_name=params["ferret_name"],
                    Brain_part=params["Brain_part"],
                    Group_name=params["Group_name"],
                )

                print(type(rec))
                print(f"Recording object: {rec}")

                if rec is not None:
                    print(f"Number of frames: {rec.get_num_frames()}")
                    print(f"Number of channels: {rec.get_num_channels()}")
                else:
                    print("No recording data found!")

                if block_qc_enabled or orientation_qc_enabled:
                    qc_src_cfg = block_qc_cfg if block_qc_enabled else orientation_qc_cfg
                    m = sample_block_qc_metrics(
                        rec,
                        block_name=block,
                        normal_phy=chan_map_info["normal_phy"] if chan_map_info is not None else None,
                        num_chunks=int(qc_src_cfg.get("num_chunks", 40)),
                        chunk_size=int(qc_src_cfg.get("chunk_size", 10000)),
                        seed=int(qc_src_cfg.get("seed", 0)),
                        flat_threshold=float(block_qc_cfg.get("flat_threshold", 1e-9)),
                        chunk_noise_factor=float(block_qc_cfg.get("chunk_noise_factor", 3.0)),
                        bad_channel_mad_factor=float(block_qc_cfg.get("bad_channel_mad_factor", 3.0)),
                        min_channel_mad=float(block_qc_cfg.get("min_channel_mad", 1.0)),
                    )
                    block_metrics.append(m)
                    powers.append(m["mean_abs_power"])
                else:
                    powers.append(compute_rec_power(rec))

                recording_list[stream].append(rec)
                block_names[stream].append(block)

            except Exception as e:

                logger.exception(f"Could not load block {block}")

        label_out_dir = output_folder / f"{params['ferret_name']}_{params['Brain_part']}__{params['Group_name']}"

        if powers:
            if block_qc_enabled:
                qc_results = evaluate_block_qc(block_metrics, cfg=block_qc_cfg)
                qc_csv = write_block_qc_csv(qc_results, label_out_dir)
                logger.info(f"Saved block QC report to {qc_csv}")
                keep_mask = [r["keep"] for r in qc_results]
                for r in qc_results:
                    msg = (
                        f"[Block QC] {r['block']}: {'KEEP' if r['keep'] else 'EXCLUDE'} "
                        f"(flat={r['flat_fraction']:.3f}, noisy_chunks={r['noisy_chunk_fraction']:.3f}, "
                        f"med_mad={r['median_mad']:.2f}, bad_ch={r['n_bad_channels']}, {r['reasons']})"
                    )
                    logger.info(msg) if r["keep"] else logger.warning(msg)
                    print(msg)
            else:
                median_power = np.median(powers)
                keep_mask = [0 < power < 2 * median_power for power in powers]

            bad_indices = [i for i, keep in enumerate(keep_mask) if not keep]

            # Save bad block names
            for i in bad_indices:
                block_name = block_names[stream][i]
                power_value = powers[i]
                log_bad_block(
                    block_name=block_name,
                    power_value=power_value,
                    output_folder=output_folder,
                    ferret_name=params["ferret_name"],
                    Brain_part=params["Brain_part"],
                    Group_name=params["Group_name"]
                )

            unflip_map = {}
            if orientation_qc_enabled and chan_map_info is not None:
                orient_report = evaluate_orientation_qc(
                    block_metrics,
                    keep_mask=keep_mask,
                    warp_ud=chan_map_info["warp_ud"],
                    cfg=orientation_qc_cfg,
                )
                orient_csv = write_orientation_qc_csv(orient_report["records"], label_out_dir)
                logger.info(f"Saved headstage orientation QC report to {orient_csv}")
                if orientation_qc_cfg.get("save_plot", True):
                    orient_png = plot_orientation_qc(orient_report, label_out_dir)
                    if orient_png is not None:
                        logger.info(f"Saved headstage orientation QC plot to {orient_png}")
                    coherence_png = plot_channel_coherence(orient_report, block_metrics, label_out_dir)
                    if coherence_png is not None:
                        logger.info(f"Saved cross-channel coherence QC plot to {coherence_png}")

                for i, r in enumerate(orient_report["records"]):
                    if r["status"] == "SKIPPED":
                        continue
                    msg = (
                        f"[Orientation QC] {r['block']}: {r['status']} "
                        f"(score_normal={r['score_normal']:.3f}, score_ud={r['score_ud']:.3f}, "
                        f"action={r['action_taken']})"
                    )
                    if r["status"] in ("UPSIDE_DOWN", "AMBIGUOUS"):
                        logger.warning(msg)
                    else:
                        logger.info(msg)
                    print(msg)
                    if r["exclude"]:
                        keep_mask[i] = False
                        log_bad_block(
                            block_name=r["block"],
                            power_value=powers[i],
                            output_folder=output_folder,
                            ferret_name=params["ferret_name"],
                            Brain_part=params["Brain_part"],
                            Group_name=params["Group_name"],
                        )
                    unflip_map[i] = bool(r["needs_unflip"])

            if block_qc_enabled or orientation_qc_enabled:
                summary_rows = build_qc_summary_table(
                    block_qc_results=qc_results if block_qc_enabled else None,
                    orientation_records=orient_report["records"] if (orientation_qc_enabled and chan_map_info is not None) else None,
                    block_names=block_names[stream],
                )
                tbl_files = write_qc_summary_tables(
                    summary_rows,
                    output_dir=label_out_dir,
                    stream=stream,
                    title=f"QC Summary: {params['ferret_name']} {params['Brain_part']} {params['Group_name']} (Stream {stream})",
                )
                logger.info(f"Saved QC summary table to {tbl_files['csv']} and {tbl_files['md']}")
                md_table_str = format_qc_summary_markdown(
                    summary_rows,
                    title=f"QC Summary: {params['ferret_name']} {params['Brain_part']} {params['Group_name']} (Stream {stream})",
                )
                print("\n" + md_table_str + "\n")
                logger.info("\n" + md_table_str)

            if args.qc_only:
                logger.info(f"QC-only mode active: completed stream {stream}; skipping session cleaning.")
                continue

            good_indices = [i for i, keep in enumerate(keep_mask) if keep]

            # Process only retained sessions, one at a time. Each result is
            # disk-backed; the concatenation below does not keep their arrays in RAM.
            # Do not delete these files while this sorting run still uses them.
            cleaned_recordings = []
            session_cache_root = (
                output_folder / "scratch" / "flat_by_session"
                / f"{params['ferret_name']}_{params['Brain_part']}__{params['Group_name']}"
                / params["rec_name"] / f"stream_{stream}" / now
            )
            for i in good_indices:
                block_name = block_names[stream][i]
                rec = recording_list[stream][i]
                perm = chan_map_info["unflip_perm"] if unflip_map.get(i, False) else None
                if perm is not None:
                    logger.info(f"Applying Warp_UD -> Warp_Chan un-flip permutation to {block_name}")
                    print(f"  Un-flipping upside-down headstage channels for {block_name}")
                logger.info(f"Removing static flat periods from {block_name}")
                cleaned = remove_static_flat_periods(
                    rec,
                    output_dir=str(session_cache_root / block_name),
                    channel_permutation=perm,
                )
                if cleaned.get_num_frames() != rec.get_num_frames():
                    raise RuntimeError(f"Sample count changed while cleaning {block_name}")
                if not np.array_equal(cleaned.get_channel_ids(), rec.get_channel_ids()):
                    raise RuntimeError(f"Channel IDs changed while cleaning {block_name}")
                cleaned_recordings.append(cleaned)
            recording_list[stream] = cleaned_recordings

        else:
            logger.warning(f"No recordings found for stream {stream}")

        if not recording_list[stream]:
            logger.error(f"No valid recordings for stream {stream}")
        else:
            logger.info(f"Found {len(recording_list[stream])} recordings for stream {stream}")
            print(f"Found {len(recording_list[stream])} recordings for stream {stream}")

    if args.qc_only:
        logger.info("QC-only run completed successfully. Exiting before concatenation and sorting.")
        print("\nQC-only run completed successfully. Exiting before concatenation and sorting.")
        return

    # Concatenate the cleaned, disk-backed sessions in their original order.
    logger.info('Concatenating recordings')

    if recording_list[stream]:
        recordings = concatenate_recordings(recording_list[stream], sampling_frequency_max_diff=1e-6)
    else:
        logger.error(f"No recordings to concatenate for stream {stream}")
        raise RuntimeError(f"No valid recordings were loaded for stream {stream}. Check datadir, rec_side, and block_range.")

    # Preprocessing Step
    logger.info('Preprocessing recordings')
    # recordings = preprocess_rec(recordings,output_folder=output_folder)
    recordings = preprocess_rec(recordings, output_dir="plots", output_folder=output_folder,
                                ferret_name=params["ferret_name"], Brain_part=params["Brain_part"],
                                Group_name=params["Group_name"])
    print(f"Preprocessed recordings: {recordings}")

    if recordings is None:
        raise ValueError("Preprocessing returned None. Check the preprocess_rec function.")

    logger.info(recordings)

    logger.info(f'start sorting with {sorter_list}')
    recording_path_stream = output_folder / "recording" / run_rec_name / f"stream_{stream}"
    Path(recording_path_stream).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving recording to {recording_path_stream}")
    recording_saved = recordings.save(folder=recording_path_stream, overwrite=True)

    if not os.path.exists(recording_path_stream):
        raise FileNotFoundError(f"Recording was not saved properly at {recording_path_stream}")

    logger.info('Sorting')

    sorter_name = sorter_list[0]
    sorter_params = dict(params.get('sorter_params', {}).get(sorter_name, {}))
    if sorter_name == 'mountainsort4':
        sorter_temp = Path(sorter_params.get('tempdir') or output_folder / 'temp').resolve()
        sorter_temp.mkdir(parents=True, exist_ok=True)
        sorter_params['tempdir'] = str(sorter_temp)

    sortings = ss.run_sorter(
        sorter_name,
        recording_saved,
        folder=working_directory,
        remove_existing_folder=True,
        **sorter_params
    )

    logger.info('Finished sorting')

    # export all to PHY
    export_all(
        sorter_list[0],
        sortings,
        recextract=recording_saved,
        rec_name=run_rec_name,
        ferret_name=params["ferret_name"],
        Brain_part=params["Brain_part"],
        Group_name=params["Group_name"],
        working_directory=working_directory,
        output_folder=output_folder,
        job_kwargs=params['job_kwargs'],
    )


if __name__ == '__main__':
    main()
