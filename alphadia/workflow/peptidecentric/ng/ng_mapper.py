"""Conversion of AlphaDIA to NG data structure and back."""

import logging

import numpy as np
import pandas as pd
from alphabase.spectral_library.flat import SpecLibFlat
from alphadia_search_rs import (
    CandidateCollection,
    CandidateFeatureCollection,
    set_num_threads,
)
from alphadia_search_rs import (
    DIAData as DiaDataNG,
)
from alphadia_search_rs import SpecLibFlat as SpecLibFlatNG

from alphadia.raw_data import DiaData

logger = logging.getLogger(__name__)


def set_ng_thread_count(thread_count: int) -> None:
    """Set the number of threads for NG computations."""
    set_num_threads(thread_count)


def dia_data_to_ng(dia_data: DiaData) -> "DiaDataNG":  # noqa: F821
    """Convert DIA data from classic to ng format."""

    spectrum_df = dia_data.spectrum_df
    peak_df = dia_data.peak_df

    cycle_len = dia_data.cycle.shape[1]
    spectrum_df_len = len(dia_data.spectrum_df)

    delta_scan_idx = np.tile(
        np.arange(cycle_len), int(spectrum_df_len / cycle_len + 1)
    )[:spectrum_df_len]
    cycle_idx = np.repeat(np.arange(int(spectrum_df_len / cycle_len + 1)), cycle_len)[
        :spectrum_df_len
    ]

    return DiaDataNG.from_arrays(
        delta_scan_idx.astype(np.int64),
        spectrum_df["isolation_lower_mz"].values.astype(np.float32),
        spectrum_df["isolation_upper_mz"].values.astype(np.float32),
        spectrum_df["peak_start_idx"].values.astype(np.int64),
        spectrum_df["peak_stop_idx"].values.astype(np.int64),
        cycle_idx.astype(np.int64),
        spectrum_df["rt"].values.astype(np.float32) * 60,
        peak_df["mz"].values.astype(np.float32),
        peak_df["intensity"].values.astype(np.float32),
        dia_data.cycle.astype(np.float32),
    )


def tims_to_ng(tims_data: "TimsTOFTranspose") -> "DiaDataNG":  # noqa: F821
    """Convert TimsTOFTranspose to DiaDataNG for the rust extraction backend.

    Collapses the ion mobility (1/K0) dimension by summing intensities across
    all scans within each MS2 frame. This produces one flat spectrum per frame,
    which the rust DIAData builder already understands. Mobility separation is
    not exploited in this implementation — week-2 work will extend the rust
    builder to handle the full 3D (RT × m/z × 1/K0) search window natively.

    Parameters
    ----------
    tims_data : TimsTOFTranspose
        Transposed timsTOF data loaded from a Bruker .d folder.

    Returns
    -------
    DiaDataNG
        Flat spectrum representation ready for the rust extraction backend.

    Notes
    -----
    RT values from alpharaw are in **seconds** (from the Bruker SQL ``Time``
    column), so no unit conversion is needed here. This differs from the mzML
    path in :func:`dia_data_to_ng`, which multiplies by 60.
    """
    scan_max = tims_data.scan_max_index
    n_frames = tims_data.frame_max_index
    cycle_len = tims_data.cycle.shape[1]

    logger.info(
        f"tims_to_ng: {n_frames} frames, {scan_max} scans/frame, "
        f"cycle_len={cycle_len}, rt range [{tims_data.rt_values[0]:.1f}, "
        f"{tims_data.rt_values[-1]:.1f}] s"
    )

    # ── Invert transposed CSR: tof_indptr/push_indices → (frame, tof, intensity) ─
    #
    # After TimsTOFTranspose.transpose():
    #   _tof_indptr  (int64,  tof_max+1):  CSR row ptrs indexed by tof_index
    #   _push_indices (uint32, n_peaks):    push = frame*scan_max + scan for each peak
    #   _intensity_values (uint16, n_peaks): reordered intensities
    #
    # We need frame_of_peak and tof_of_peak, then group-by-frame.

    push_indices = tims_data._push_indices          # uint32 (n_peaks,)
    tof_indptr   = tims_data._tof_indptr            # int64  (tof_max+1,)
    intensity    = tims_data._intensity_values      # uint16 (n_peaks,)
    mz_values    = tims_data.mz_values              # float64 (tof_max,)
    n_peaks      = len(push_indices)

    frame_of_peak = (push_indices // scan_max).astype(np.int64)

    # Reconstruct tof_index per peak by inverting tof_indptr.
    # np.repeat is ~4x faster than a Python loop over 400k tof indices.
    tof_of_peak = np.repeat(
        np.arange(len(tof_indptr) - 1, dtype=np.int32),
        np.diff(tof_indptr),
    )

    # Sort peaks by (frame, tof) so we can slice per-frame cheaply
    sort_order   = np.lexsort((tof_of_peak, frame_of_peak))
    sorted_frame = frame_of_peak[sort_order]
    sorted_tof   = tof_of_peak[sort_order]
    sorted_int   = intensity[sort_order].astype(np.float32)

    # Boundaries of each frame's peaks in the sorted array
    frame_boundaries = np.searchsorted(sorted_frame, np.arange(n_frames + 1))

    # ── Build per-frame spectra, summing duplicate tof indices ───────────────
    frame_peak_mz  = [None] * n_frames
    frame_peak_int = [None] * n_frames

    for f in range(n_frames):
        sl   = slice(int(frame_boundaries[f]), int(frame_boundaries[f + 1]))
        tofs = sorted_tof[sl]
        ints = sorted_int[sl]

        if len(tofs) == 0:
            frame_peak_mz[f]  = np.empty(0, dtype=np.float32)
            frame_peak_int[f] = np.empty(0, dtype=np.float32)
            continue

        # Sum intensities at the same tof index (multiple mobility scans → one peak)
        unique_tofs, inverse = np.unique(tofs, return_inverse=True)
        summed_int = np.zeros(len(unique_tofs), dtype=np.float32)
        np.add.at(summed_int, inverse, ints)

        frame_peak_mz[f]  = mz_values[unique_tofs].astype(np.float32)
        frame_peak_int[f] = summed_int

    # ── Flatten to contiguous peak arrays with start/stop indices ────────────
    peak_counts    = np.array([len(x) for x in frame_peak_mz], dtype=np.int64)
    peak_start_idx = np.empty(n_frames, dtype=np.int64)
    peak_start_idx[0] = 0
    if n_frames > 1:
        peak_start_idx[1:] = np.cumsum(peak_counts[:-1])
    peak_stop_idx = peak_start_idx + peak_counts

    all_mz  = np.concatenate(frame_peak_mz).astype(np.float32)
    all_int = np.concatenate(frame_peak_int).astype(np.float32)

    logger.info(
        f"tims_to_ng: flattened to {len(all_mz):,} peaks across {n_frames} frames"
    )

    # ── Isolation windows per frame from dia_mz_cycle ────────────────────────
    #
    # dia_mz_cycle[i] = [lower_mz, upper_mz] for cycle position i.
    # MS2 frames map to cycle positions; MS1 frames get (0, 0).
    dia_mz       = tims_data.dia_mz_cycle           # (cycle_len, 2)
    frames_table = tims_data.frames                  # pd.DataFrame from SQL
    ms2_mask     = frames_table["MsMsType"].values != 0  # bool (n_frames,)

    isolation_lower = np.zeros(n_frames, dtype=np.float32)
    isolation_upper = np.zeros(n_frames, dtype=np.float32)

    frame_in_cycle = np.arange(n_frames) % cycle_len
    # Guard against dia_mz_cycle being shorter than max cycle position
    valid_cycle_pos = frame_in_cycle[ms2_mask]
    in_bounds = valid_cycle_pos < len(dia_mz)
    ms2_idx = np.where(ms2_mask)[0]

    isolation_lower[ms2_idx[in_bounds]] = dia_mz[valid_cycle_pos[in_bounds], 0].astype(
        np.float32
    )
    isolation_upper[ms2_idx[in_bounds]] = dia_mz[valid_cycle_pos[in_bounds], 1].astype(
        np.float32
    )

    # ── Cycle-level indices ───────────────────────────────────────────────────
    delta_scan_idx = (np.arange(n_frames) % cycle_len).astype(np.int64)
    cycle_idx      = (np.arange(n_frames) // cycle_len).astype(np.int64)

    # rt_values from alpharaw is already in seconds (Bruker SQL Time column)
    rt_seconds = tims_data.rt_values.astype(np.float32)

    return DiaDataNG.from_arrays(
        delta_scan_idx,
        isolation_lower,
        isolation_upper,
        peak_start_idx,
        peak_stop_idx,
        cycle_idx,
        rt_seconds,
        all_mz,
        all_int,
        tims_data.cycle.astype(np.float32),
    )


def speclib_to_ng(
    speclib: SpecLibFlat,
    *,
    rt_column: str,
    precursor_mz_column: str,
    fragment_mz_column: str,
) -> "SpecLibFlatNG":  # noqa: F821
    """Convert speclib from classic to ng format."""

    precursor_df = speclib.precursor_df
    fragment_df = speclib.fragment_df

    return SpecLibFlatNG.from_arrays(
        precursor_df["precursor_idx"].values.astype(np.uint64),
        precursor_df["mz_library"].values.astype(np.float32),
        precursor_df[precursor_mz_column].values.astype(np.float32),
        precursor_df["rt_library"].values.astype(np.float32),
        precursor_df[rt_column].values.astype(np.float32),
        precursor_df["nAA"].values.astype(np.uint8),
        precursor_df["flat_frag_start_idx"].values.astype(np.uint64),
        precursor_df["flat_frag_stop_idx"].values.astype(np.uint64),
        fragment_df["mz_library"].values.astype(np.float32),
        fragment_df[fragment_mz_column].values.astype(np.float32),
        fragment_df["intensity"].values.astype(np.float32),
        fragment_df["cardinality"].values.astype(np.uint8),
        fragment_df["charge"].values.astype(np.uint8),
        fragment_df["loss_type"].values.astype(np.uint8),
        fragment_df["number"].values.astype(np.uint8),
        fragment_df["position"].values.astype(np.uint8),
        fragment_df["type"].values.astype(np.uint8),
    )


def get_feature_names() -> list[str]:
    """Get feature names from NG CandidateFeatureCollection."""
    blacklist = ["fwhm_rt"]  # TODO: remove
    return [
        f for f in CandidateFeatureCollection.get_feature_names() if f not in blacklist
    ]


def parse_candidates(
    candidates: CandidateCollection, spectral_library: SpecLibFlat, dia_data: DiaDataNG
) -> pd.DataFrame:
    """Parse candidates from NG to classic format."""

    cycle_len = dia_data.cycle.shape[1]

    result = candidates.to_arrays()

    precursor_idx = result[0]
    rank = result[1]
    score = result[2]
    scan_center = result[3]
    scan_start = result[4]
    scan_stop = result[5]
    frame_center = result[6]
    frame_start = result[7]
    frame_stop = result[8]

    candidates_df = pd.DataFrame(
        {
            "precursor_idx": precursor_idx,
            "rank": rank,
            "score": score,
            "scan_center": scan_center,
            "scan_start": scan_start,
            "scan_stop": scan_stop,
            "frame_center": frame_center,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )

    candidates_df = candidates_df.merge(
        spectral_library.precursor_df[["precursor_idx", "elution_group_idx", "decoy"]],
        on="precursor_idx",
        how="left",
    )

    candidates_df["frame_start"] = candidates_df["frame_start"] * cycle_len
    candidates_df["frame_stop"] = candidates_df["frame_stop"] * cycle_len
    candidates_df["frame_center"] = candidates_df["frame_center"] * cycle_len

    candidates_df["scan_start"] = 0
    candidates_df["scan_stop"] = 1
    candidates_df["scan_center"] = 0

    return candidates_df


def candidates_to_ng(
    candidates_df: pd.DataFrame, dia_data: DiaDataNG
) -> CandidateCollection:
    """Convert candidates from classic to NG format."""

    cycle_len = dia_data.cycle.shape[1]

    candidates = CandidateCollection.from_arrays(
        candidates_df["precursor_idx"].values.astype(np.uint64),
        candidates_df["rank"].values.astype(np.uint64),
        candidates_df["score"].values.astype(np.float32),
        candidates_df["scan_center"].values.astype(np.uint64),
        candidates_df["scan_start"].values.astype(np.uint64),
        candidates_df["scan_stop"].values.astype(np.uint64),
        candidates_df["frame_center"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_start"].values.astype(np.uint64) // cycle_len,
        candidates_df["frame_stop"].values.astype(np.uint64) // cycle_len,
    )
    return candidates


def to_features_df(
    candidate_features: CandidateFeatureCollection, spectral_library: SpecLibFlat
) -> pd.DataFrame:
    """Convert NG candidate features to classic format."""

    features_dict = candidate_features.to_dict_arrays()

    features_df = pd.DataFrame(features_dict)

    features_df = features_df.merge(
        spectral_library.precursor_df[
            [
                "precursor_idx",
                "decoy",
                "elution_group_idx",
                "channel",
                "proteins",
            ]
        ],
        on="precursor_idx",
        how="left",
    )

    features_df.rename(columns={"fwhm_rt": "cycle_fwhm"}, inplace=True)

    return features_df


def parse_quantification(
    quantified_speclib: "SpecLibFlatQuantified",  # noqa: F821
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert NG quantified spectral library to classic precursor and fragments DataFrame."""

    precursor_dict, fragment_dict = quantified_speclib.to_dict_arrays()

    precursor_df = pd.DataFrame(precursor_dict).rename(
        columns={"idx": "precursor_idx"}
    )  # TODO: remove when #96 is merged

    fragments_df = pd.DataFrame(fragment_dict).rename(
        columns={
            "correlation_observed": "correlation",
            "mass_error_observed": "mass_error",
        }
    )

    return precursor_df, fragments_df
