"""
Parity tests: CUDA center-head postprocessing vs PyTorch reference.

Runs in CI (CPU, stubs skipped) and on the dev machine (full CUDA).
Asserts max absolute difference < TOLERANCE for peak locations and
decoded box coordinates.
"""

import numpy as np
import pytest

TOLERANCE = 1e-3


# ── PyTorch reference implementations ────────────────────────────────────────

def ref_peak_finding(heatmap: np.ndarray, radius: int, threshold: float):
    """
    Pure NumPy reference: max-pool NMS on a [C, H, W] heatmap.
    Returns list of (row, col, class_id, score) tuples.
    """
    # TODO: implement with scipy.ndimage.maximum_filter or manual sliding window
    raise NotImplementedError


def ref_box_decode(reg_maps: np.ndarray, peaks: list, cfg: dict) -> np.ndarray:
    """
    Pure NumPy reference box decoding.
    Returns array of shape [N, 10] with decoded box parameters.
    """
    # TODO: implement the decode equations from box_decode_kernel comments
    raise NotImplementedError


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPeakFinding:

    def test_single_obvious_peak(self):
        """One dominant peak — CUDA and NumPy must agree on location."""
        pytest.skip("CUDA kernel not yet implemented")

    def test_adjacent_peaks_suppressed(self):
        """Two adjacent peaks: only the higher one should survive."""
        pytest.skip("CUDA kernel not yet implemented")

    def test_below_threshold_suppressed(self):
        """Peaks below score_threshold must be excluded."""
        pytest.skip("CUDA kernel not yet implemented")


class TestBoxDecode:

    def test_zero_offset_identity(self):
        """Zero regression offsets → box centres at voxel-grid coordinates."""
        pytest.skip("CUDA kernel not yet implemented")

    def test_size_decode_log_space(self):
        """log-space size regression → exp gives positive dimensions."""
        pytest.skip("CUDA kernel not yet implemented")
