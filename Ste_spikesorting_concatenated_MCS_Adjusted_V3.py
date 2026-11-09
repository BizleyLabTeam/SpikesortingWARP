### REC A = Right, REC B = Left


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
from spikesorting_scripts.preprocessing import remove_disconnection_events


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
    padding_samples=20, probe=None, output_dir="preprocessed_recording"
):
    if recording is None:
        raise ValueError("Input recording is None")

    if recording.get_num_segments() != 1:
        raise ValueError("Flat-period cleaning expects one session with one segment")

    sampling_frequency = recording.get_sampling_frequency()
    min_flat_samples = int(min_flat_duration_s * sampling_frequency)

    traces = recording.get_traces()
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


############################### MAIN #######################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="path to the json file containing the parameters")
    args = parser.parse_args()

    with open(args.params_file) as json_file:
        minified = jsmin(json_file.read())
        params = json.loads(minified)
    output_folder = Path(params['output_folder']).resolve()
    logpath = Path(params['logpath'])
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

    datadir = Path(params['datadir']) / params[
        'rec_name']  ## Provide the path of the 'datadir', including the folder that is two level above the final folder containing the .h5 file
    print(datadir)  ##'rec_name' is the path to the folder before the final folder that contains the .h5 file
    streams = params[
        'streams']  ## 'streams' is the path to the stream related to your data recording, containing values (channels * frames) in .h5 groups

    working_directory = Path(params['working_directory']) / params['rec_name']
    working_directory.mkdir(parents=True, exist_ok=True)

    blocks = [bl.name for bl in datadir.glob('Block_J*')]
    print(blocks)
    blocks.sort(key=lambda f: int(re.sub('\D', '', f)))
    pbar = tqdm(blocks)

    recording_list = {stream: [] for stream in streams}

    for stream in streams:
        powers = []
        rec_list = []
        logger.info(f'Loading stream {stream}')
        block_names = {stream: [] for stream in streams}

        for block in pbar:
            pbar.set_postfix_str(f'loading {block}')
            logger.info(f'Loading block {block}')
            print(f"Loading block {block}")

            try:
                h5_files = list((datadir / block).rglob("*.h5"))
                if len(h5_files) != 1:
                    logger.error(f"Expected 1 .h5 in {datadir / block}, found {len(h5_files)}: {h5_files}")
                    continue

                h5_file = h5_files[0]

                # --- load recording ---
                rec = se.read_mcsh5(str(h5_file), stream_id=stream)

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

                powers.append(compute_rec_power(rec))
                recording_list[stream].append(rec)
                block_names[stream].append(block)

            except Exception as e:

                logger.exception(f"Could not load block {block}")

        #        if powers:
        #        # only keep recordings with power below 2*median and above 0
        #            median_power = np.median(powers)
        #            recording_list[stream] = [recording_list[stream][i] for i, power in enumerate(powers) if 0 < power < 2 * median_power]
        #        else:
        #            logger.warning(f"No recordings found for stream {stream}")
        #
        bad_blocks = []

        if powers:
            median_power = np.median(powers)
            good_indices = [i for i, power in enumerate(powers) if 0 < power < 2 * median_power]
            bad_indices = [i for i in range(len(powers)) if i not in good_indices]

            # Save bad block names
            for i in bad_indices:
                block_name = block_names[stream][i]  # or however you access it
                power_value = powers[i]
                log_bad_block(
                    block_name=block_name,
                    power_value=power_value,
                    output_folder=output_folder,
                    ferret_name=params["ferret_name"],
                    Brain_part=params["Brain_part"],
                    Group_name=params["Group_name"]
                )

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
                logger.info(f"Removing static flat periods from {block_name}")
                cleaned = remove_static_flat_periods(
                    rec, output_dir=str(session_cache_root / block_name)
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

    # Concatenate the cleaned, disk-backed sessions in their original order.
    logger.info('Concatenating recordings')

    if recording_list[stream]:

        recordings = concatenate_recordings(recording_list[stream], sampling_frequency_max_diff=1e-6)

    else:

        logger.error(f"No recordings to concatenate for stream {stream}")

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
    recording_path_stream = output_folder / "recording" / params["rec_name"] / f"stream_{stream}"
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
        rec_name=params["rec_name"],
        ferret_name=params["ferret_name"],
        Brain_part=params["Brain_part"],
        Group_name=params["Group_name"],
        working_directory=working_directory,
        output_folder=output_folder,
        job_kwargs=params['job_kwargs'],
    )


if __name__ == '__main__':
    main()
