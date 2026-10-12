import numpy as np
import pytest

from biosaur2 import cutils, cfast


def frame(mz, it, im):
    o = np.argsort(mz, kind='stable')
    return {'m/z array': np.asarray(mz, float)[o], 'intensity array': np.asarray(it, np.float32)[o],
            'mean inverse reduced ion mobility array': np.asarray(im, float)[o]}


def centroid_both(z, **kw):
    a = dict(mz_step=8e-6 * 1500, hill_mz_accuracy=8, ion_mobility_accuracy=0.05, pasefmini=0, pasefminlh=1)
    a.update(kw)
    args = (a['mz_step'], a['hill_mz_accuracy'], a['ion_mobility_accuracy'], a['pasefmini'], a['pasefminlh'])
    ref = [np.array(x) for x in cutils.centroid_pasef_scan(dict(z), *args)]
    fast = cfast.centroid_pasef_scan_fast(dict(z), *args)
    assert fast is not None
    for r, f in zip(ref, fast):
        assert np.array_equal(r, f)
    return ref


def test_interleaved_ions_keep_all_intensity():
    # R2: ion B (other IM) lies between two peaks of ion A; it used to be skipped
    mz = [500.0000, 500.0010, 500.0016, 500.0030]
    it = [100, 50, 80, 40]
    im = [1.00, 1.30, 1.01, 1.02]
    mzs, its, ims = centroid_both(frame(mz, it, im))
    assert np.isclose(its.sum(), sum(it))
    assert len(its) == 2
    assert np.isclose(its[np.argmin(np.abs(ims - 1.3))], 50)


def test_peak_close_to_centroid_is_absorbed():
    # R2: a peak within tolerance of the cluster centroid but not of the chain's last peak joins the cluster
    mz = [600.0000, 600.0040, 600.0020]
    it = [100, 100, 10]
    im = [1.00, 1.00, 1.06]     # third peak: > 0.05 from the chain peaks' IM? no: 0.06 from both
    mzs, its, ims = centroid_both(frame(mz, it, im), ion_mobility_accuracy=0.065)
    assert np.isclose(its.sum(), 210)


def test_random_frames_fast_equals_cutils():
    rng = np.random.default_rng(1)
    for _ in range(5):
        n = 3000
        mz = np.round(rng.uniform(400, 410, n), 3)       # many exact m/z ties
        it = np.round(10 ** rng.uniform(1, 3, n))
        im = rng.choice(np.linspace(0.8, 1.3, 200), n)
        centroid_both(frame(mz, it, im), pasefmini=50)
