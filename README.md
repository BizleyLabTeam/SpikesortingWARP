# WARP spike sorting with MountainSort4

This repository contains the V3 (Zheyi's third version since Nadia gave the original one she's been using) pipeline for MCS HDF5 recordings and the helper modules it imports. The environment file pins Python 3.11.4 and the runtime package versions from the working Windows `spikesortingWARP` environment, including SpikeInterface 0.102.3 and MountainSort4 1.0.7.

## Set up the environment

Clone the repository and open a terminal in its root folder. You can build the environment using either **`uv`** (fastest) or **Conda / Mamba**.

### Option A: Using `uv` (Fastest)

[uv](https://github.com/astral-sh/uv) creates and installs the environment in seconds using `requirements.txt`:

```powershell
# 1. Create a virtual environment pinned to Python 3.11
uv venv --python 3.11 .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install the dependencies
uv pip install -r requirements.txt
```

> **VS Code Tip:** After building with `uv`, press `Ctrl+Shift+P` in VS Code, choose **Python: Select Interpreter**, and select `.\.venv\Scripts\python.exe`.

---

### Option B: Using Conda or Mamba

If you prefer using Anaconda, Miniconda, or Mamba, build from `environment.yml`:

```powershell
# Using Conda
conda env create --file environment.yml
conda activate spikesortingWARP

# Or using Mamba (faster Conda alternative)
mamba env create --file environment.yml
mamba activate spikesortingWARP
```

---

> **Note on Windows C++ Build Tools:** On Windows, compiling `isosplit5` from source requires the free *Microsoft C++ Build Tools* if your system does not already have a compatible C/C++ compiler installed.

## Set up a recording

Use `Stephen_spikesorting_params_concatenated_MCS_BB.json` as a worked example. It shows the settings and paths from Zheyi's run. Copy it to `config.local.json` and edit the paths and recording labels for your machine.

```powershell
Copy-Item -LiteralPath Stephen_spikesorting_params_concatenated_MCS_BB.json -Destination config.local.json
```

Set `datadir` to the folder containing your recording collections and `rec_name` to the collection you want to sort (or set `datadir` directly to the folder containing `Block_J*` directories).

The pipeline now supports **two folder layouts**:
1. **Pre-split directories (Zheyi and Nadia's preferred option):** Separate folders for Rec A and Rec B where each `Block_J*` folder contains a single `.h5` file.
2. **Combined directories (standard block layout, Rebecca's preference):** A single folder per block (e.g. `Z:/WARPData/F2206_Hurricane/Block_J1-1128`) containing both Rec A and Rec B `.h5` files (such as `*Rec_A.h5`, `*Rec_B_2.h5`, `*Rec_B_3.h5`). Set `"rec_side": "A"` or `"rec_side": "B"` in the JSON (or pass `--rec-side A` / `--rec-side B` on the command line) to sort one side at a time.

```text
data/
  F2206_Hurricane/
    Block_J1-1128/
      2026-08-05T17-03-08Rec_A.h5
      2026-08-05T17-03-08Rec_B_2.h5
```

| JSON field | What to enter |
| --- | --- |
| `datadir`, `rec_name` | The input base folder and its recording collection (or direct block parent folder). |
| `rec_side` *(optional)* | `"A"` or `"B"` (also accepts `"Rec_A"`, `"Rec_B"`, `"Rec_B_2"`, etc.). Selects which side's `.h5` file to load when both A and B are in the same block folder. If omitted, inferred from `rec_name`/`Brain_part` or defaults to the single `.h5` file in pre-split folders. |
| `block_range` *(optional)* | `[start, end]` block numbers (inclusive), e.g. `[1100, 1130]`, or `null` to process all `Block_J*` folders. |
| `streams` | One numeric MCS analog stream ID, such as `[1]`. Select the stream with the 32 electrode channels (in MCS W2100 HDF5 files, stream 1 is `Electrode Raw Data` with 32 channels, while stream 0 is 1-channel `Quality Data`). |
| `ferret_name`, `Brain_part`, `Group_name` | Labels used to organize CSV and Phy output. |
| `output_folder`, `logpath`, `working_directory` | Writable destinations for results, logs, and sorter working files. |
| `block_qc` *(optional)* | Configures pre-concatenation wireless dropout & noise block QC (`enabled`, `max_flat_fraction`: e.g. 0.70, `max_noisy_chunk_fraction`, `max_power_ratio`, `min_mad`, `max_excess_bad_channels`: e.g. 10). Saves `block_qc_report.csv` and unified `qc_summary_table.csv` / `.md`. |
| `orientation_qc` *(optional)* | Configures headstage orientation QC across blocks using `docs/Warp_to_WirelessHeadstage_ChanMap.txt` (`enabled`, `reference_block`: e.g. `"906"` or `"Block_J1-906"`, `action`: `"warn"`, `"flip"`, or `"exclude"`, `save_plot`). Saves `headstage_orientation_qc.csv`, summary bar plot `headstage_orientation_qc.png`, cross-channel spatial correlation & coherence plot `headstage_channel_coherence.png`, and unified `qc_summary_table.csv` / `.md`. |

Relative paths are resolved from the terminal's current folder. Keep the terminal in the repository root, or use absolute paths in the JSON. Use forward slashes for Windows paths in the JSON, for example `"D:/Recordings"`.



## Run the pipeline

```powershell
python Ste_spikesorting_concatenated_MCS_Adjusted_V3.py config.local.json
```

You can also toggle or override the optional QC steps from the command line without editing your JSON:

```powershell
python Ste_spikesorting_concatenated_MCS_Adjusted_V3.py config.local.json --block-qc --orientation-qc --orientation-action flip --reference-block 906
```

To run **just the diagnostic QC** across your blocks and export the summary tables and orientation plot without running the full spike sorting:

```powershell
python Ste_spikesorting_concatenated_MCS_Adjusted_V3.py config.local.json --qc-only --reference-block 906
```

The script creates the log and working folders as needed. Under the configured output folder, it saves flat-period-cleaned sessions in `scratch/`, the preprocessed sorting input in `recording/`, and the CSV and Phy results in the animal/brain/group folder. MountainSort4 uses a run subfolder under `temp/` by default and cleans up that subfolder when sorting finishes. The session cache stays on disk.

The Phy configuration is saved at `<output_folder>/<ferret_name>_<Brain_part>__<Group_name>/<rec_name>/mountainsort4/phy/params.py`. Open it with `phy template-gui` in an environment where Phy is installed.

> **Note on Phy:** Phy does not play nicely with many other packages. Make a separate virtual environment just for Phy, DO NOT TRY TO INSTALL IT IN THIS ONE.

Use a fresh output location when you want to keep an earlier run's Phy curation. A rerun targeting the same location replaces the waveform and Phy export folders.

## Share the repository

Share the repository URL with lab members who have access. They supply their own recordings and local JSON. The default data and output folders and local configurations are ignored by Git.

The bundled helpers come from [Jules Lebert's spikesorting_scripts](https://github.com/JulesLebert/spikesorting_scripts). Their license is included in `spikesorting_scripts/LICENSE`.
