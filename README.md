# MCS spike sorting with MountainSort4

This repository contains the V3 (Zheyi's third version since Nadia gave the original one she's been using) pipeline for MCS HDF5 recordings and the helper modules it imports. The environment file pins Python 3.11.4 and the runtime package versions from the working Windows `spikesortingWARP` environment, including SpikeInterface 0.102.3 and MountainSort4 1.0.7.

## Set up the environment

Clone the repository and open a Conda-enabled PowerShell terminal in its root folder. Create the environment once, then activate it each time you run the pipeline.

```powershell
conda env create --file environment.yml
conda activate spikesortingWARP
```

If you already have the matching environment, run the activation command. On Windows, a source build of `isosplit5` requires Microsoft C++ Build Tools if pip reports that it cannot find a compiler.

## Set up a recording

Copy the example configuration for local use.

```powershell
Copy-Item -LiteralPath config.example.json -Destination config.local.json
```

Set `datadir` to the folder containing your recording collections and `rec_name` to the collection you want to sort. Each `Block_J*` folder must contain exactly one `.h5` file, which can be in a subfolder. The example expects this layout.

```text
data/
  Recording_01/
    Block_J1-001/
      recording.h5
    Block_J1-002/
      recording.h5
```

| JSON field | What to enter |
| --- | --- |
| `datadir`, `rec_name` | The input base folder and its recording collection. 
| `streams` | One numeric MCS analog stream ID, such as `[0]` or `[1]`. Select the stream that matches the 32-channel WARP probe mapping. [0] is recB and [1] is recA
| `ferret_name`, `Brain_part`, `Group_name` | Labels used to organize CSV and Phy output. 
| `output_folder`, `logpath`, `working_directory` | Writable destinations for results, logs, and sorter working files. 

I also put my own JSON file in as an example (Stephen_spikesorting_params_concatenated_MCS_BB)

Relative paths are resolved from the terminal's current folder. Keep the terminal in the repository root, or use absolute paths in the JSON. Use forward slashes for Windows paths in the JSON, for example `"D:/Recordings"`.



## Run the pipeline

```powershell
python Ste_spikesorting_concatenated_MCS_Adjusted_V3.py config.local.json
```

The script creates the log and working folders as needed. Under the configured output folder, it saves flat-period-cleaned sessions in `scratch/`, the preprocessed sorting input in `recording/`, and the CSV and Phy results in the animal/brain/group folder. MountainSort4 uses a run subfolder under `temp/` by default and cleans up that subfolder when sorting finishes. The session cache stays on disk.

The example's Phy configuration is at `outputs/run_01/Animal_01_Right__Set01/Recording_01/mountainsort4/phy/params.py`. Open it with `phy template-gui` in an environment where Phy is installed.

Use a fresh output location when you want to keep an earlier run's Phy curation. A rerun targeting the same location replaces the waveform and Phy export folders.

## Share the repository

Share the repository URL with lab members who have access. They supply their own recordings and local JSON. The default data and output folders and local configurations are ignored by Git.

The bundled helpers come from [Jules Lebert's spikesorting_scripts](https://github.com/JulesLebert/spikesorting_scripts). Their license is included in `spikesorting_scripts/LICENSE`.
