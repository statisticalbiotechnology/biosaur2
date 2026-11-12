"""-hill_max_missing: a hill may bridge one scan in which its peak is absent."""
import numpy as np
import pytest

from biosaur2 import cutils, cfast

MZ_STEP = 8e-6 * 1500


def frames(peaks_per_frame, with_im=False):
    """peaks_per_frame: one list of (mz, intensity[, im]) per frame."""
    out = []
    for peaks in peaks_per_frame:
        peaks = sorted(peaks)
        z = {'m/z array': np.array([p[0] for p in peaks], dtype=np.float64),
             'intensity array': np.array([p[1] for p in peaks], dtype=np.float32)}
        if with_im:
            z['mean inverse reduced ion mobility array'] = np.array([p[2] for p in peaks], dtype=np.float64)
        out.append(z)
    return out


def partition(hills):
    """Hill membership as a tuple of first-peak labels, independent of the ids."""
    first = {}
    return tuple(first.setdefault(h, i) for i, h in enumerate(np.asarray(hills).tolist()))


def both(data, max_missing, paseftol=0.0):
    ref, _ = cutils.detect_hills(data, {'htol': 8, 'hill_max_missing': max_missing}, MZ_STEP, paseftol)
    fast = cfast.detect_hills_fast(data, 8, MZ_STEP, paseftol, 1, False, max_missing)
    assert fast is not None
    assert partition(ref['hills_idx_array']) == partition(fast[0]['hills_idx_array'])
    return partition(ref['hills_idx_array'])


# an ion at m/z 600 that drops out of frame 3, plus an unrelated ion that does not
DROPOUT = [[(600.0, 100), (700.0, 50)],
           [(600.0, 120), (700.0, 55)],
           [(600.0, 140), (700.0, 60)],
           [(700.0, 65)],
           [(600.0, 130), (700.0, 60)],
           [(600.0, 110), (700.0, 55)]]


def ion_600_hills(part, data):
    labels, i = [], 0
    for z in data:
        for mz in z['m/z array']:
            if abs(mz - 600.0) < 0.01:
                labels.append(part[i])
            i += 1
    return set(labels)


def test_gap_ends_hill_by_default():
    data = frames(DROPOUT)
    assert len(ion_600_hills(both(data, 0), data)) == 2


def test_gap_bridged_with_max_missing_1():
    data = frames(DROPOUT)
    assert len(ion_600_hills(both(data, 1), data)) == 1


def test_two_missing_scans_are_not_bridged():
    data = frames([[(600.0, 100)], [(600.0, 120)], [(700.0, 50)], [(700.0, 50)], [(600.0, 110)]])
    assert len(ion_600_hills(both(data, 1), data)) == 2


def test_continued_peak_is_not_bridged_again():
    # frame 2 holds the ion and a weaker peak 3 ppm away: the ion continues the
    # hill from frame 1, and the weaker peak must not reach back to frame 0,
    # whose peak frame 1 already continued. A peak at m/z 500 keeps the ion off
    # index 0 of each frame, which detect_hills never marks as taken (known
    # issue #5, deliberately left as it is).
    data = frames([[(500.0, 50), (600.0, 100)],
                   [(500.0, 50), (600.0, 120)],
                   [(500.0, 50), (600.0, 140), (600.0018, 10)]])
    part = both(data, 1)
    ion_0, ion_2, weak_2 = 1, 5, 6
    assert part[ion_2] == part[ion_0]
    assert part[weak_2] != part[ion_0]


def test_ion_mobility_tolerance_applies_across_the_gap():
    near = frames([[(600.0, 100, 1.00)], [(600.0, 120, 1.00)], [(700.0, 5, 0.8)],
                   [(600.0, 110, 1.02)]], with_im=True)
    far = frames([[(600.0, 100, 1.00)], [(600.0, 120, 1.00)], [(700.0, 5, 0.8)],
                  [(600.0, 110, 1.20)]], with_im=True)
    assert len(ion_600_hills(both(near, 1, paseftol=0.05), near)) == 1
    assert len(ion_600_hills(both(far, 1, paseftol=0.05), far)) == 2


def test_gaps_fast_equals_fallback(synth, run_pipeline):
    assert run_pipeline(synth, fast=True, hill_max_missing=1) == run_pipeline(synth, fast=False, hill_max_missing=1)


def test_default_output_unchanged(synth, run_pipeline):
    assert run_pipeline(synth, fast=True) == run_pipeline(synth, fast=True, hill_max_missing=0)
