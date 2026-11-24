"""Block-level quality control (wireless dropout/noise) and headstage orientation QC.

Provides two optional, configurable pre-concatenation QC steps for WARP MCS .h5
recordings:
1. Block-level wireless dropout & noise QC (`evaluate_block_qc`):
   Inspects sampled chunks from each recording block for excessive flat-lining
   (wireless dropouts), high-amplitude disconnection bursts, dead/noisy channels,
   and abnormal overall noise floor / power, excluding corrupted blocks before
   flat-period cleaning and concatenation.
2. Headstage orientation QC (`evaluate_orientation_qc`):
   Uses the MCS-to-WARP channel map (`docs/Warp_to_WirelessHeadstage_ChanMap.txt`)
   to compare each block's 32-channel noise-floor profile (log-MAD) and
   within-block spatial correlation matrix against a reference block under both
   normal (`Warp_Chan`) and upside-down (`Warp_UD`) mappings. Supports flagging
   (`warn`), excluding (`exclude`), or automatically un-flipping (`flip`)
   upside-down blocks prior to concatenation.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spikeinterface.core import get_random_data_chunks

logger = logging.getLogger(__name__)


def load_warp_channel_map(chan_map_path: str | Path) -> dict[str, np.ndarray]:
    """Load MCS-to-WARP channel mapping and derive normal/upside-down permutations.

    Parameters
    ----------
    chan_map_path : str or Path
        Path to tab-delimited `Warp_to_WirelessHeadstage_ChanMap.txt`.

    Returns
    -------
    dict with keys:
        - `normal_phy`: 0-based MCS index for each `Warp_Chan` 0..31.
        - `ud_phy`: 0-based MCS index when headstage is upside down (`Warp_UD`).
        - `warp_ud`: `Warp_UD` index for each `Warp_Chan` 0..31.
        - `unflip_perm`: permutation of length 32 such that `traces[:, unflip_perm]`
          remaps an upside-down MCS recording back to normal MCS channel order.
    """
    path = Path(chan_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Channel map file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"Warp_Chan", "MCS_Chan", "Warp_UD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Channel map {path} missing columns: {sorted(missing)}")

    df = df.sort_values("Warp_Chan").reset_index(drop=True)
    if "Phy_Chan" in df.columns:
        phy_col = df["Phy_Chan"].to_numpy(dtype=int)
    else:
        phy_col = df["MCS_Chan"].to_numpy(dtype=int) - 1

    warp_chan = df["Warp_Chan"].to_numpy(dtype=int)
    warp_ud = df["Warp_UD"].to_numpy(dtype=int)

    warp_to_phy = dict(zip(warp_chan, phy_col))
    normal_phy = np.array([warp_to_phy[w] for w in warp_chan], dtype=int)
    ud_phy = np.array([warp_to_phy[w_ud] for w_ud in warp_ud], dtype=int)

    unflip_perm = np.zeros(len(warp_chan), dtype=int)
    unflip_perm[normal_phy] = ud_phy

    return {
        "normal_phy": normal_phy,
        "ud_phy": ud_phy,
        "warp_ud": warp_ud,
        "unflip_perm": unflip_perm,
    }


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation ignoring non-finite pairs."""
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3:
        return 0.0
    av = a[valid]
    bv = b[valid]
    sa = np.std(av)
    sb = np.std(bv)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def sample_block_qc_metrics(
    recording,
    block_name: str,
    normal_phy: np.ndarray | None = None,
    num_chunks: int = 40,
    chunk_size: int = 10000,
    seed: int = 0,
    flat_threshold: float = 1e-9,
    chunk_noise_factor: float = 3.0,
    bad_channel_mad_factor: float = 3.0,
    min_channel_mad: float = 1.0,
) -> dict[str, Any]:
    """Sample random chunks from one block and compute noise, dropout, and spatial metrics."""
    num_frames = int(recording.get_num_frames())
    num_channels = int(recording.get_num_channels())

    if num_frames <= 1:
        return {
            "block": block_name,
            "num_frames": num_frames,
            "num_channels": num_channels,
            "mean_abs_power": 0.0,
            "median_mad": 0.0,
            "flat_fraction": 1.0,
            "noisy_chunk_fraction": 0.0,
            "n_bad_channels": num_channels,
            "bad_channel_fraction": 1.0,
            "warp_mad": None,
            "warp_corr_matrix": None,
        }

    eff_chunk_size = min(int(chunk_size), max(2, num_frames // 2))
    max_chunks = max(1, num_frames // eff_chunk_size)
    eff_num_chunks = min(int(num_chunks), max_chunks)

    chunks = get_random_data_chunks(
        recording,
        num_chunks_per_segment=eff_num_chunks,
        chunk_size=eff_chunk_size,
        seed=seed,
    )

    mean_abs_power = float(np.mean(np.abs(chunks)))

    # Reshape into (eff_num_chunks, eff_chunk_size, num_channels)
    usable = (chunks.shape[0] // eff_chunk_size) * eff_chunk_size
    c3 = chunks[:usable].reshape(-1, eff_chunk_size, num_channels)

    # Flat-line fraction across transitions
    flat_fraction = float(np.mean(np.abs(np.diff(c3, axis=1)) < flat_threshold))

    # Chunk-level MAD per channel
    c3_med = np.median(c3, axis=1, keepdims=True)
    chunk_mad = np.median(np.abs(c3 - c3_med), axis=1) / 0.6745  # (n_chunks, n_ch)
    ch_chunk_med_mad = np.median(chunk_mad, axis=0)
    # Guard zero median MAD with a tiny floor so dead channels aren't flagged as bursty
    denom = np.maximum(ch_chunk_med_mad, 1e-6)
    noisy_chunk_fraction = float(np.mean(chunk_mad > chunk_noise_factor * denom[None, :]))

    # Overall per-channel MAD noise floor
    global_med = np.median(chunks, axis=0)
    mad_per_ch = np.median(np.abs(chunks - global_med), axis=0) / 0.6745
    median_mad = float(np.median(mad_per_ch))

    dead_mask = mad_per_ch < min_channel_mad
    noisy_ch_mask = (
        mad_per_ch > bad_channel_mad_factor * max(median_mad, 1e-6)
        if median_mad > 0
        else np.zeros(num_channels, dtype=bool)
    )
    bad_ch_mask = dead_mask | noisy_ch_mask
    n_bad_channels = int(np.sum(bad_ch_mask))
    bad_channel_fraction = float(n_bad_channels / max(num_channels, 1))

    warp_mad = None
    warp_corr_matrix = None
    if normal_phy is not None and num_channels == len(normal_phy):
        warp_traces = chunks[:, normal_phy].astype(np.float64)
        warp_mad = mad_per_ch[normal_phy].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_mat = np.corrcoef(warp_traces.T)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)
        warp_corr_matrix = corr_mat

    return {
        "block": block_name,
        "num_frames": num_frames,
        "num_channels": num_channels,
        "mean_abs_power": mean_abs_power,
        "median_mad": median_mad,
        "flat_fraction": flat_fraction,
        "noisy_chunk_fraction": noisy_chunk_fraction,
        "n_bad_channels": n_bad_channels,
        "bad_channel_fraction": bad_channel_fraction,
        "warp_mad": warp_mad,
        "warp_corr_matrix": warp_corr_matrix,
    }


def evaluate_block_qc(
    metrics_list: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate block-level dropout & noise QC across all loaded blocks."""
    cfg = cfg or {}
    max_power_ratio = float(cfg.get("max_power_ratio", 2.0))
    min_mad = float(cfg.get("min_mad", 1.0))
    max_flat_fraction = float(cfg.get("max_flat_fraction", 0.70))
    max_noisy_chunk_fraction = float(cfg.get("max_noisy_chunk_fraction", 0.30))
    # Relative bad channel check: max allowed excess bad channels over surrounding baseline
    max_excess_bad_channels = cfg.get("max_excess_bad_channels", 10)
    if max_excess_bad_channels is not None:
        max_excess_bad_channels = int(max_excess_bad_channels)
    # Absolute bad channel threshold is disabled by default (None) so blocks with few good
    # channels are preserved when moving electrodes.
    max_bad_channel_fraction = cfg.get("max_bad_channel_fraction", None)
    if max_bad_channel_fraction is not None:
        max_bad_channel_fraction = float(max_bad_channel_fraction)

    positive_powers = [m["mean_abs_power"] for m in metrics_list if m["mean_abs_power"] > 0]
    global_median_power = float(np.median(positive_powers)) if positive_powers else 0.0

    all_bad_ch = [m["n_bad_channels"] for m in metrics_list]
    cohort_median_bad = float(np.median(all_bad_ch)) if all_bad_ch else 0.0

    results: list[dict[str, Any]] = []
    for i, m in enumerate(metrics_list):
        reasons: list[str] = []
        pwr = m["mean_abs_power"]
        med_mad = m["median_mad"]
        flat_f = m["flat_fraction"]
        noisy_f = m["noisy_chunk_fraction"]
        bad_ch_f = m["bad_channel_fraction"]
        n_bad = m["n_bad_channels"]

        if pwr <= 0:
            reasons.append(f"mean_abs_power={pwr:.3g}<=0")
        elif global_median_power > 0 and pwr >= max_power_ratio * global_median_power:
            reasons.append(
                f"power_ratio={pwr / global_median_power:.2f}>={max_power_ratio:.2f}"
            )

        if med_mad < min_mad:
            reasons.append(f"median_mad={med_mad:.2f}<{min_mad:.2f}")
        if flat_f > max_flat_fraction:
            reasons.append(f"flat_fraction={flat_f:.3f}>{max_flat_fraction:.3f}")
        if noisy_f > max_noisy_chunk_fraction:
            reasons.append(f"noisy_chunk_fraction={noisy_f:.3f}>{max_noisy_chunk_fraction:.3f}")

        # Baseline for bad channels: compare against local window (+- 2 blocks) or cohort median
        neighbor_indices = [
            j for j in range(max(0, i - 2), min(len(metrics_list), i + 3))
            if j != i
        ]
        if neighbor_indices:
            local_baseline_bad = float(np.median([metrics_list[j]["n_bad_channels"] for j in neighbor_indices]))
        else:
            local_baseline_bad = cohort_median_bad

        excess_bad = int(n_bad - local_baseline_bad)
        if max_excess_bad_channels is not None and excess_bad >= max_excess_bad_channels:
            reasons.append(
                f"excess_bad_channels={n_bad} vs baseline {local_baseline_bad:.0f} (+{excess_bad}>={max_excess_bad_channels})"
            )

        if max_bad_channel_fraction is not None and bad_ch_f > max_bad_channel_fraction:
            reasons.append(f"bad_channel_fraction={bad_ch_f:.3f}>{max_bad_channel_fraction:.3f}")

        keep = len(reasons) == 0
        results.append(
            {
                "block": m["block"],
                "keep": keep,
                "reasons": "; ".join(reasons) if reasons else "PASS",
                "mean_abs_power": pwr,
                "global_median_power": global_median_power,
                "median_mad": med_mad,
                "flat_fraction": flat_f,
                "noisy_chunk_fraction": noisy_f,
                "n_bad_channels": n_bad,
                "local_baseline_bad": local_baseline_bad,
                "excess_bad_channels": excess_bad,
                "bad_channel_fraction": bad_ch_f,
            }
        )
    return results


def evaluate_orientation_qc(
    metrics_list: list[dict[str, Any]],
    keep_mask: list[bool],
    warp_ud: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare each block's channel noise-floor & spatial correlation against a reference block.

    Returns a dictionary containing `reference_block` and per-block `records`.
    """
    cfg = cfg or {}
    min_corr = float(cfg.get("min_corr", 0.50))
    min_margin = float(cfg.get("min_margin", 0.10))
    action = str(cfg.get("action", "warn")).lower()
    exclude_ambiguous = bool(cfg.get("exclude_ambiguous", False))
    requested_ref = cfg.get("reference_block")

    valid_indices = [
        i
        for i, (m, keep) in enumerate(zip(metrics_list, keep_mask))
        if keep and m.get("warp_mad") is not None and m.get("warp_corr_matrix") is not None
    ]

    if not valid_indices:
        return {"reference_block": None, "records": []}

    ref_idx = valid_indices[0]
    if requested_ref is not None:
        req_str = str(requested_ref).strip()
        matched_idx = None
        # 1. Exact string match
        for i in valid_indices:
            if str(metrics_list[i]["block"]) == req_str:
                matched_idx = i
                break
        # 2. Token match (split by hyphen/underscore, e.g. "906" in "Block_J1-906")
        if matched_idx is None:
            for i in valid_indices:
                b_name = str(metrics_list[i]["block"])
                tokens = [t for t in re.split(r"[-_]", b_name) if t]
                if req_str in tokens or b_name.endswith(f"-{req_str}") or b_name.endswith(f"_{req_str}"):
                    matched_idx = i
                    break
        # 3. Substring match
        if matched_idx is None:
            for i in valid_indices:
                if req_str in str(metrics_list[i]["block"]):
                    matched_idx = i
                    break

        if matched_idx is not None:
            ref_idx = matched_idx
        else:
            # Check if requested block exists among invalid/excluded blocks
            all_names = [str(m["block"]) for m in metrics_list]
            found_in_all = any(
                req_str == name or req_str in re.split(r"[-_]", name) or req_str in name
                for name in all_names
            )
            if found_in_all:
                logger.warning(
                    f"Requested reference_block '{requested_ref}' was excluded by Block QC; "
                    f"falling back to '{metrics_list[ref_idx]['block']}'."
                )
            else:
                logger.warning(
                    f"Requested reference_block '{requested_ref}' not among valid blocks; "
                    f"using '{metrics_list[ref_idx]['block']}' instead."
                )

    ref_m = metrics_list[ref_idx]
    ref_block = ref_m["block"]
    ref_log_mad = np.log1p(ref_m["warp_mad"])
    ref_corr_mat = ref_m["warp_corr_matrix"]
    iu = np.triu_indices(len(warp_ud), k=1)
    ref_spatial_vec = ref_corr_mat[iu]

    records: list[dict[str, Any]] = []
    for i, m in enumerate(metrics_list):
        if not keep_mask[i] or m.get("warp_mad") is None or m.get("warp_corr_matrix") is None:
            records.append(
                {
                    "block": m["block"],
                    "status": "SKIPPED",
                    "action_taken": "excluded_by_block_qc" if not keep_mask[i] else "no_chan_map",
                    "needs_unflip": False,
                    "exclude": not keep_mask[i],
                    "mad_corr_normal": np.nan,
                    "mad_corr_ud": np.nan,
                    "spatial_corr_normal": np.nan,
                    "spatial_corr_ud": np.nan,
                    "score_normal": np.nan,
                    "score_ud": np.nan,
                    "median_mad": m["median_mad"],
                }
            )
            continue

        b_log_mad = np.log1p(m["warp_mad"])
        b_log_mad_ud = b_log_mad[warp_ud]

        mad_corr_norm = _safe_pearson(ref_log_mad, b_log_mad)
        mad_corr_ud = _safe_pearson(ref_log_mad, b_log_mad_ud)

        b_corr_mat = m["warp_corr_matrix"]
        b_corr_mat_ud = b_corr_mat[np.ix_(warp_ud, warp_ud)]

        spatial_corr_norm = _safe_pearson(ref_spatial_vec, b_corr_mat[iu])
        spatial_corr_ud = _safe_pearson(ref_spatial_vec, b_corr_mat_ud[iu])

        # Spatial cross-channel covariance captures the physical shank geometry.
        # When the spatial correlation matrix decisively indicates an orientation
        # (difference >= min_margin and max correlation >= min_corr), it takes precedence over
        # single-channel noise floor (MAD) artifacts.
        # We also weight spatial correlation higher (75% spatial, 25% MAD) in the blended score.
        score_norm = 0.75 * spatial_corr_norm + 0.25 * mad_corr_norm
        score_ud = 0.75 * spatial_corr_ud + 0.25 * mad_corr_ud

        if i == ref_idx:
            status = "REFERENCE"
        elif (spatial_corr_ud - spatial_corr_norm) >= min_margin and spatial_corr_ud >= min_corr:
            status = "UPSIDE_DOWN"
        elif (spatial_corr_norm - spatial_corr_ud) >= min_margin and spatial_corr_norm >= min_corr:
            status = "NORMAL"
        elif score_norm >= min_corr and (score_norm - score_ud) >= min_margin:
            status = "NORMAL"
        elif score_ud >= min_corr and (score_ud - score_norm) >= min_margin:
            status = "UPSIDE_DOWN"
        else:
            status = "AMBIGUOUS"

        needs_unflip = False
        exclude = False
        action_taken = "none"

        if status == "UPSIDE_DOWN":
            if action == "flip":
                needs_unflip = True
                action_taken = "unflipped_to_reference"
            elif action == "exclude":
                exclude = True
                action_taken = "excluded"
            else:
                action_taken = "warned_only"
        elif status == "AMBIGUOUS":
            if action == "exclude" and exclude_ambiguous:
                exclude = True
                action_taken = "excluded_ambiguous"
            else:
                action_taken = "warned_ambiguous"

        records.append(
            {
                "block": m["block"],
                "status": status,
                "action_taken": action_taken,
                "needs_unflip": needs_unflip,
                "exclude": exclude,
                "mad_corr_normal": mad_corr_norm,
                "mad_corr_ud": mad_corr_ud,
                "spatial_corr_normal": spatial_corr_norm,
                "spatial_corr_ud": spatial_corr_ud,
                "score_normal": score_norm,
                "score_ud": score_ud,
                "median_mad": m["median_mad"],
            }
        )

    return {"reference_block": ref_block, "records": records}


def write_block_qc_csv(results: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write block-level dropout & noise QC report to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "block_qc_report.csv"
    fieldnames = [
        "block",
        "keep",
        "reasons",
        "mean_abs_power",
        "global_median_power",
        "median_mad",
        "flat_fraction",
        "noisy_chunk_fraction",
        "n_bad_channels",
        "local_baseline_bad",
        "excess_bad_channels",
        "bad_channel_fraction",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})
    return csv_path


def write_orientation_qc_csv(records: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write headstage orientation QC report to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "headstage_orientation_qc.csv"
    fieldnames = [
        "block",
        "status",
        "action_taken",
        "needs_unflip",
        "exclude",
        "mad_corr_normal",
        "mad_corr_ud",
        "spatial_corr_normal",
        "spatial_corr_ud",
        "score_normal",
        "score_ud",
        "median_mad",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in fieldnames})
    return csv_path


def build_qc_summary_table(
    block_qc_results: list[dict[str, Any]] | None = None,
    orientation_records: list[dict[str, Any]] | None = None,
    block_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a unified summary table merging block QC and headstage orientation QC.

    Returns a list of dicts with formatted metrics ready for CSV/Markdown export.
    """
    bqc_by_name = {r["block"]: r for r in (block_qc_results or [])}
    orient_by_name = {r["block"]: r for r in (orientation_records or [])}

    all_blocks = []
    if block_names is not None:
        all_blocks = list(block_names)
    else:
        seen = set()
        for r in (block_qc_results or []):
            if r["block"] not in seen:
                seen.add(r["block"])
                all_blocks.append(r["block"])
        for r in (orientation_records or []):
            if r["block"] not in seen:
                seen.add(r["block"])
                all_blocks.append(r["block"])

    def _fmt_pair(a: Any, b: Any, precision: int = 3) -> str:
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            return "-"
        return f"{a:.{precision}f} / {b:.{precision}f}"

    summary_rows: list[dict[str, Any]] = []
    for b in all_blocks:
        bqc = bqc_by_name.get(b)
        ori = orient_by_name.get(b)

        if bqc is not None:
            block_qc_status = "PASS" if bqc.get("keep", False) else "EXCLUDE"
            reasons = bqc.get("reasons", "PASS")
            flat_frac = bqc.get("flat_fraction", np.nan)
            noisy_frac = bqc.get("noisy_chunk_fraction", np.nan)
            med_mad = bqc.get("median_mad", np.nan)
            bad_ch = bqc.get("n_bad_channels", 0)
        else:
            block_qc_status = "N/A"
            reasons = "N/A"
            flat_frac = np.nan
            noisy_frac = np.nan
            med_mad = ori.get("median_mad", np.nan) if ori else np.nan
            bad_ch = np.nan

        if ori is not None:
            orient_status = ori.get("status", "N/A")
            action = ori.get("action_taken", "none")
            mad_norm = ori.get("mad_corr_normal", np.nan)
            mad_ud = ori.get("mad_corr_ud", np.nan)
            spat_norm = ori.get("spatial_corr_normal", np.nan)
            spat_ud = ori.get("spatial_corr_ud", np.nan)
            score_norm = ori.get("score_normal", np.nan)
            score_ud = ori.get("score_ud", np.nan)
        else:
            orient_status = "N/A"
            action = "none"
            mad_norm = np.nan
            mad_ud = np.nan
            spat_norm = np.nan
            spat_ud = np.nan
            score_norm = np.nan
            score_ud = np.nan

        summary_rows.append(
            {
                "block": b,
                "block_qc": block_qc_status,
                "reasons": reasons,
                "flat_fraction": flat_frac,
                "noisy_chunk_fraction": noisy_frac,
                "median_mad": med_mad,
                "n_bad_channels": bad_ch,
                "orientation": orient_status,
                "action_taken": action,
                "mad_corr_normal": mad_norm,
                "mad_corr_ud": mad_ud,
                "spatial_corr_normal": spat_norm,
                "spatial_corr_ud": spat_ud,
                "score_normal": score_norm,
                "score_ud": score_ud,
                "mad_norm_vs_ud": _fmt_pair(mad_norm, mad_ud),
                "spatial_norm_vs_ud": _fmt_pair(spat_norm, spat_ud),
                "score_norm_vs_ud": _fmt_pair(score_norm, score_ud),
            }
        )

    return summary_rows


def format_qc_summary_markdown(
    summary_rows: list[dict[str, Any]],
    title: str | None = None,
) -> str:
    """Format the QC summary records as a GitHub/Markdown table string."""
    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")

    headers = [
        "Block",
        "Block QC",
        "`flat_frac`",
        "`noisy_frac`",
        "`med_mad`",
        "Orientation",
        "`mad_norm` / `mad_ud`",
        "`spatial_norm` / `spatial_ud`",
        "`score_norm` / `score_ud`",
        "Action",
    ]
    align = [":---", ":---", ":---:", ":---:", ":---:", ":---", ":---:", ":---:", ":---:", ":---"]

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(align) + " |")

    for r in summary_rows:
        b = f"`{r['block']}`"
        bqc = f"`{r['block_qc']}`" if r["block_qc"] != "N/A" else "N/A"
        ff = f"{r['flat_fraction']:.4f}" if np.isfinite(r["flat_fraction"]) else "-"
        nf = f"{r['noisy_chunk_fraction']:.4f}" if np.isfinite(r["noisy_chunk_fraction"]) else "-"
        mm = f"{r['median_mad']:.2f}" if np.isfinite(r["median_mad"]) else "-"

        ori_raw = r["orientation"]
        if ori_raw in ("REFERENCE", "NORMAL"):
            ori = f"**{ori_raw}**"
        elif ori_raw == "UPSIDE_DOWN":
            ori = f"**`{ori_raw}`**"
        elif ori_raw in ("AMBIGUOUS", "SKIPPED"):
            ori = f"*{ori_raw}*"
        else:
            ori = ori_raw

        mad_pair = f"`{r['mad_norm_vs_ud']}`" if r["mad_norm_vs_ud"] != "-" else "-"
        spat_pair = f"`{r['spatial_norm_vs_ud']}`" if r["spatial_norm_vs_ud"] != "-" else "-"
        score_pair = f"`{r['score_norm_vs_ud']}`" if r["score_norm_vs_ud"] != "-" else "-"
        act = r["action_taken"]

        row_vals = [b, bqc, ff, nf, mm, ori, mad_pair, spat_pair, score_pair, act]
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def write_qc_summary_tables(
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    stream: int | str | None = None,
    title: str | None = None,
) -> dict[str, Path]:
    """Export unified QC summary as both CSV and Markdown files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_stream_{stream}" if stream is not None else ""

    csv_path = output_dir / f"qc_summary_table{suffix}.csv"
    md_path = output_dir / f"qc_summary_table{suffix}.md"

    fieldnames = [
        "block",
        "block_qc",
        "reasons",
        "flat_fraction",
        "noisy_chunk_fraction",
        "median_mad",
        "n_bad_channels",
        "orientation",
        "action_taken",
        "mad_corr_normal",
        "mad_corr_ud",
        "spatial_corr_normal",
        "spatial_corr_ud",
        "score_normal",
        "score_ud",
        "mad_norm_vs_ud",
        "spatial_norm_vs_ud",
        "score_norm_vs_ud",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    md_content = format_qc_summary_markdown(
        summary_rows,
        title=title or (f"QC Summary (Stream {stream})" if stream is not None else "QC Summary Table"),
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    # Refresh the base-name summaries with the current stream's results.
    base_csv = output_dir / "qc_summary_table.csv"
    base_md = output_dir / "qc_summary_table.md"
    if stream is not None:
        with open(base_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        with open(base_md, "w", encoding="utf-8") as f:
            f.write(md_content + "\n")

    return {"csv": csv_path, "md": md_path}


def plot_orientation_qc(
    orientation_report: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Save a diagnostic bar plot comparing Normal vs Upside-Down scores per block."""
    records = [r for r in orientation_report.get("records", []) if r["status"] != "SKIPPED"]
    if not records:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "headstage_orientation_qc.png"

    blocks = [r["block"] for r in records]
    s_norm = [r["score_normal"] for r in records]
    s_ud = [r["score_ud"] for r in records]
    statuses = [r["status"] for r in records]

    x = np.arange(len(blocks))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(blocks)), 4.6))
    ax.bar(x - width / 2, s_norm, width, label="Normal (Warp_Chan)", color="#2563eb", alpha=0.88)
    ax.bar(x + width / 2, s_ud, width, label="Upside-Down (Warp_UD)", color="#ea580c", alpha=0.88)

    for xi, st, sn, su in zip(x, statuses, s_norm, s_ud):
        top = max(sn, su, 0.0) + 0.04
        col = "#16a34a" if st in ("REFERENCE", "NORMAL") else ("#dc2626" if st == "UPSIDE_DOWN" else "#d97706")
        ax.text(xi, top, st, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=col)

    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(-0.4, 1.18)
    ax.set_ylabel("Mean Correlation Score (log-MAD & Spatial)", fontsize=9.5)
    ax.set_title(
        f"Headstage Orientation QC (Reference: {orientation_report.get('reference_block')})",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_channel_coherence(
    orientation_report: dict[str, Any],
    block_metrics: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Save a diagnostic multi-panel plot with an NxN block spatial correlation matrix and 32x32 channel heatmaps."""
    valid_blocks = [
        m for m in block_metrics
        if m.get("warp_corr_matrix") is not None
    ]
    if len(valid_blocks) < 2:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "headstage_channel_coherence.png"

    status_map = {r["block"]: r["status"] for r in orientation_report.get("records", [])}

    n_blocks = len(valid_blocks)
    block_names = [m["block"] for m in valid_blocks]
    mats = [m["warp_corr_matrix"] for m in valid_blocks]

    # Compute NxN cross-block spatial correlation matrix
    n_ch = mats[0].shape[0]
    iu = np.triu_indices(n_ch, k=1)
    block_conf = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            v1 = mats[i][iu]
            v2 = mats[j][iu]
            s1 = np.std(v1)
            s2 = np.std(v2)
            if s1 > 1e-6 and s2 > 1e-6:
                block_conf[i, j] = np.corrcoef(v1, v2)[0, 1]
            else:
                block_conf[i, j] = 0.0

    # Grid layout: left is NxN correlation matrix; right is grid of 32x32 heatmaps
    n_cols = min(4, max(1, n_blocks))
    n_rows = (n_blocks + n_cols - 1) // n_cols

    fig_w = max(11.0, 4.8 + 2.5 * n_cols)
    fig_h = max(5.2, 2.6 * n_rows + 0.8)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols + 2, width_ratios=[2.2, 0.3] + [1.0] * n_cols)

    # 1. Left subplot: Cross-Block Spatial Correlation Matrix
    ax_conf = fig.add_subplot(gs[:, 0])
    im_conf = ax_conf.imshow(block_conf, cmap="RdBu_r", vmin=-0.2, vmax=1.0, origin="upper")
    cbar_conf = plt.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
    cbar_conf.set_label("Spatial Correlation", fontsize=8.5)

    short_labels = []
    for b in block_names:
        st = status_map.get(b, "")
        short_tag = "REF" if st == "REFERENCE" else ("UD" if st == "UPSIDE_DOWN" else ("NORM" if st == "NORMAL" else st))
        b_short = b.replace("Block_J1-", "").replace("Block_", "")
        short_labels.append(f"{b_short} ({short_tag})")

    ax_conf.set_xticks(range(n_blocks))
    ax_conf.set_yticks(range(n_blocks))
    ax_conf.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax_conf.set_yticklabels(short_labels, fontsize=8)
    ax_conf.set_title("Cross-Block Spatial Correlation Matrix", fontweight="bold", fontsize=10.5, pad=10)

    if n_blocks <= 16:
        for i in range(n_blocks):
            for j in range(n_blocks):
                val = block_conf[i, j]
                col = "white" if val > 0.55 or val < -0.1 else "black"
                ax_conf.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=col)

    # 2. Right subplots: 32x32 channel heatmaps for each block
    for idx, (b, C) in enumerate(zip(block_names, mats)):
        r_idx = idx // n_cols
        c_idx = (idx % n_cols) + 2
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.imshow(C, cmap="viridis", vmin=-0.2, vmax=1.0, origin="upper")
        st = status_map.get(b, "UNKNOWN")
        color = "#16a34a" if st in ("REFERENCE", "NORMAL") else ("#dc2626" if st == "UPSIDE_DOWN" else "#d97706")
        b_short = b.replace("Block_J1-", "").replace("Block_", "")
        ax.set_title(f"{b_short} ({st})", fontsize=8.5, fontweight="bold", color=color)

        if r_idx == n_rows - 1:
            ax.set_xlabel("Ch (0-31)", fontsize=7.5)
        else:
            ax.set_xticklabels([])

        if c_idx == 2:
            ax.set_ylabel("Ch (0-31)", fontsize=7.5)
        else:
            ax.set_yticklabels([])

        ax.tick_params(labelsize=6.5)

    ref_name = orientation_report.get("reference_block")
    fig.suptitle(
        f"Headstage Cross-Channel Coherence & Spatial Correlation (Reference: {ref_name})",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path
