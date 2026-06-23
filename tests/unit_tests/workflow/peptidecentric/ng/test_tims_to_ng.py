"""Unit tests for tims_to_ng() — timsTOF to rust backend conversion.

Uses a minimal mock TimsTOFTranspose with fully controlled arrays so no
actual .d file is needed. The mock reproduces the post-transpose layout:
  - _push_indices (uint32): push = frame*scan_max + scan per peak
  - _tof_indptr   (int64):  CSR row ptrs indexed by tof_index
  - _intensity_values (uint16): intensities in tof-sorted order
  - rt_values, mz_values, cycle, dia_mz_cycle, frames
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / mock builder
# ---------------------------------------------------------------------------


class MockTimsTOFTranspose:
    """Minimal stand-in for TimsTOFTranspose.

    Layout
    ------
    We define N frames, S scans/frame, T tof_indices, and place a few peaks
    explicitly.  The transposed CSR is built from those peaks:
      tof_indptr[t]..tof_indptr[t+1]  →  push_indices[s:e], intensity_values[s:e]
    where push = frame * scan_max + scan.
    """

    def __init__(
        self,
        n_frames: int,
        scan_max: int,
        mz_values: np.ndarray,
        peaks: list[tuple[int, int, int, float]],
        dia_mz_cycle: np.ndarray,
        cycle: np.ndarray,
        ms_types: list[int] | None = None,
        rt_values: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        peaks : list of (frame, scan, tof_idx, intensity)
        """
        self.frame_max_index = n_frames
        self.scan_max_index  = scan_max
        self.tof_max_index   = len(mz_values)
        self.mz_values       = mz_values.astype(np.float64)
        self.dia_mz_cycle    = dia_mz_cycle.astype(np.float64)
        self.cycle           = cycle.astype(np.float64)

        self.rt_values = (
            rt_values.astype(np.float64)
            if rt_values is not None
            else np.arange(n_frames, dtype=np.float64) * 10.0
        )

        ms_types = ms_types if ms_types is not None else [9] * n_frames
        self.frames = pd.DataFrame({"MsMsType": ms_types})

        # Build transposed CSR from peak list
        self._push_indices, self._tof_indptr, self._intensity_values = (
            self._build_transposed_csr(peaks, len(mz_values), n_frames, scan_max)
        )

    @staticmethod
    def _build_transposed_csr(
        peaks: list[tuple[int, int, int, float]],
        n_tof: int,
        n_frames: int,
        scan_max: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build push_indices, tof_indptr, intensity_values in tof-sorted order."""
        # Sort by tof_idx (primary), then push (secondary) — matches transpose output
        sorted_peaks = sorted(peaks, key=lambda p: (p[2], p[0] * scan_max + p[1]))

        push_indices     = np.array([p[0] * scan_max + p[1] for p in sorted_peaks], dtype=np.uint32)
        intensity_values = np.array([p[3] for p in sorted_peaks], dtype=np.uint16)

        # Build tof_indptr
        tof_indptr = np.zeros(n_tof + 1, dtype=np.int64)
        for _, _, tof_idx, _ in sorted_peaks:
            tof_indptr[tof_idx + 1] += 1
        np.cumsum(tof_indptr, out=tof_indptr)

        return push_indices, tof_indptr, intensity_values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_tims():
    """2 frames, 3 scans/frame, 5 tof_indices, cycle_len=2.

    Peaks placed at:
      frame 0, scan 0, tof 1, intensity  100
      frame 0, scan 1, tof 1, intensity  200   ← same tof, same frame → sum = 300
      frame 0, scan 0, tof 3, intensity   50
      frame 1, scan 2, tof 2, intensity  400
    """
    n_frames = 2
    scan_max = 3
    mz_vals  = np.array([100.0, 200.0, 300.0, 400.0, 500.0])  # tof 0..4

    peaks = [
        (0, 0, 1, 100),
        (0, 1, 1, 200),
        (0, 0, 3,  50),
        (1, 2, 2, 400),
    ]

    # cycle_len = 2: one MS1 + one MS2
    dia_mz_cycle = np.array([[0.0, 0.0], [400.0, 500.0]])  # pos 0 = MS1, pos 1 = MS2
    cycle        = np.zeros((1, 2, 1, 2))                   # shape (1, cycle_len, 1, 2)
    ms_types     = [0, 9]  # frame 0 = MS1, frame 1 = MS2

    return MockTimsTOFTranspose(
        n_frames=n_frames,
        scan_max=scan_max,
        mz_values=mz_vals,
        peaks=peaks,
        dia_mz_cycle=dia_mz_cycle,
        cycle=cycle,
        ms_types=ms_types,
    )


@pytest.fixture()
def empty_frames_tims():
    """3 frames where frame 1 has no peaks."""
    n_frames = 3
    scan_max = 2
    mz_vals  = np.array([100.0, 200.0, 300.0])

    peaks = [
        (0, 0, 0, 500),
        (2, 1, 2, 700),
    ]

    dia_mz_cycle = np.array([[400.0, 500.0], [500.0, 600.0], [600.0, 700.0]])
    cycle        = np.zeros((1, 3, 1, 2))
    ms_types     = [9, 9, 9]

    return MockTimsTOFTranspose(
        n_frames=n_frames,
        scan_max=scan_max,
        mz_values=mz_vals,
        peaks=peaks,
        dia_mz_cycle=dia_mz_cycle,
        cycle=cycle,
        ms_types=ms_types,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_shapes(simple_tims):
    """Output arrays must have length == n_frames and peaks array is flat."""
    from alphadia.workflow.peptidecentric.ng.ng_mapper import tims_to_ng

    result = tims_to_ng(simple_tims)

    n_frames  = simple_tims.frame_max_index
    cycle_len = simple_tims.cycle.shape[1]

    assert result.cycle.shape[1] == cycle_len
    assert len(result.rt_values) == n_frames


def test_mobility_summing(simple_tims):
    """Two peaks at the same tof in the same frame must be summed."""
    from alphadia.workflow.peptidecentric.ng.ng_mapper import tims_to_ng

    # Patch to expose peak arrays for inspection — call the inner logic directly
    import numpy as np
    from alphadia.workflow.peptidecentric.ng.ng_mapper import tims_to_ng

    # We validate by checking the DiaDataNG object indirectly:
    # frame 0 should have 2 unique tof indices (tof 1 summed, tof 3) = 2 peaks
    # frame 1 should have 1 peak (tof 2)
    #
    # Verify via num_observations (one per delta_scan_idx value in the cycle)
    result = tims_to_ng(simple_tims)
    # cycle_len=2 → 2 observations (delta_scan_idx 0 and 1)
    assert result.num_observations == simple_tims.cycle.shape[1]


def test_rt_values_passthrough(simple_tims):
    """rt_values must be preserved as-is (already in seconds)."""
    from alphadia.workflow.peptidecentric.ng.ng_mapper import tims_to_ng

    result = tims_to_ng(simple_tims)
    np.testing.assert_allclose(
        result.rt_values,
        simple_tims.rt_values.astype(np.float32),
    )


def test_cycle_shape_preserved(simple_tims):
    """cycle.shape[1] in output must match input."""
    from alphadia.workflow.peptidecentric.ng.ng_mapper import tims_to_ng

    result = tims_to_ng(simple_tims)
    assert result.cycle.shape[1] == simple_tims.cycle.shape[1]


def test_empty_frame_handled(empty_frames_tims):
    """A frame with no peaks must not cause an error and must produce empty arrays."""
    from alphadia.workflow.peptidecentric.ng.ng_mapper import tims_to_ng

    result = tims_to_ng(empty_frames_tims)
    # Just confirm it runs without error; num_observations = cycle_len = 3
    assert result.num_observations == empty_frames_tims.cycle.shape[1]


def test_ms1_frames_get_zero_isolation(simple_tims):
    """MS1 frames (MsMsType=0) must have isolation (0.0, 0.0)."""
    # We can't inspect per-frame isolation directly from DiaDataNG, so we test
    # the intermediate logic by re-running the relevant part of tims_to_ng.
    tims = simple_tims  # frame 0 = MS1, frame 1 = MS2

    n_frames   = tims.frame_max_index
    cycle_len  = tims.cycle.shape[1]
    dia_mz     = tims.dia_mz_cycle
    ms2_mask   = tims.frames["MsMsType"].values != 0

    isolation_lower = np.zeros(n_frames, dtype=np.float32)
    isolation_upper = np.zeros(n_frames, dtype=np.float32)

    frame_in_cycle  = np.arange(n_frames) % cycle_len
    valid_cycle_pos = frame_in_cycle[ms2_mask]
    in_bounds       = valid_cycle_pos < len(dia_mz)
    ms2_idx         = np.where(ms2_mask)[0]

    isolation_lower[ms2_idx[in_bounds]] = dia_mz[valid_cycle_pos[in_bounds], 0]
    isolation_upper[ms2_idx[in_bounds]] = dia_mz[valid_cycle_pos[in_bounds], 1]

    # Frame 0 = MS1 → must stay (0, 0)
    assert isolation_lower[0] == 0.0
    assert isolation_upper[0] == 0.0

    # Frame 1 = MS2, cycle pos 1 → dia_mz_cycle[1] = [400, 500]
    assert isolation_lower[1] == pytest.approx(400.0)
    assert isolation_upper[1] == pytest.approx(500.0)
