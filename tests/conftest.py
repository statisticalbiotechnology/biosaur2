"""Small synthetic timsTOF-like MS1 data and a helper that runs the default
feature detection pipeline (biosaur2.main.process_file) on it, with the fast
path on or off.  No external data needed."""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEUTRON = 1.00335
PROTON = 1.0072765


def make_data(n_frames=24, n_peptides=60, noise_per_frame=1500, seed=0, frame_dt=1.1, distort=None, mass_range=(800, 3000)):
    """List of spectrum dicts as utils.process_mzml returns them (combined ion
    mobility frames: one ion = many (m/z, 1/K0) points, integer intensities)."""
    from scipy.stats import binom
    rng = np.random.default_rng(seed)
    total_rt = n_frames * frame_dt
    mass = rng.uniform(mass_range[0], mass_range[1], n_peptides)
    charge = rng.choice([1, 2, 3], size=n_peptides, p=[0.2, 0.6, 0.2])
    mz0 = (mass + charge * PROTON) / charge
    keep = (mz0 > 360) & (mz0 < 1450)
    mass, charge, mz0 = mass[keep], charge[keep], mz0[keep]
    rt_apex = rng.uniform(0.2 * total_rt, 0.8 * total_rt, len(mass))
    rt_sigma = rng.uniform(2.0, 5.0, len(mass))
    im_apex = rng.uniform(0.75, 1.35, len(mass))
    apex_int = 10 ** rng.uniform(3.5, 5.0, len(mass))
    im_grid = np.linspace(1.6, 0.6, 600)
    patterns = []
    for m in mass:
        p = binom.pmf(np.arange(6), int(round(m / 111.1254 * 4.9384)), 0.0107)
        p = p / p.max()
        if distort is not None:      # (isotope index, factor): a pattern averagine does not explain
            p[distort[0]] *= distort[1]
        patterns.append(p)
    data = []
    for f in range(n_frames):
        rt = f * frame_dt
        mzs, its, ims = [], [], []
        w = np.exp(-0.5 * ((rt - rt_apex) / rt_sigma) ** 2)
        for p in np.flatnonzero(w > 0.02):
            scans = im_grid[np.abs(im_grid - im_apex[p]) <= 0.03]
            imw = np.exp(-0.5 * ((scans - im_apex[p]) / 0.01) ** 2)
            imw /= imw.sum()
            for k, rel in enumerate(patterns[p]):
                if rel < 0.05:
                    continue
                cnt = rng.poisson(apex_int[p] * w[p] * rel * imw).astype(float)
                sel = cnt > 0
                if not sel.any():
                    continue
                mzt = mz0[p] + k * NEUTRON / charge[p]
                mzs.append(mzt * (1 + rng.normal(0, 3e-6, sel.sum())))
                its.append(cnt[sel] * 10)
                ims.append(scans[sel])
        nn = rng.poisson(noise_per_frame)
        mzs.append(rng.uniform(350, 1500, nn))
        its.append(np.round(10 ** rng.normal(1.3, 0.3, nn)))
        ims.append(rng.choice(im_grid, nn))
        mz = np.concatenate(mzs); it = np.concatenate(its); im = np.concatenate(ims)
        sel = it >= 1
        mz, it, im = mz[sel], it[sel], im[sel]
        o = np.argsort(mz, kind='stable')
        data.append({
            'ms level': 1,
            'm/z array': mz[o].astype(np.float64),
            'intensity array': it[o].astype(np.float32),
            'mean inverse reduced ion mobility array': im[o].astype(np.float64),
            'scanList': {'scan': [{'scan start time': 100.0 + rt}]},
            'index': f,
            'id': 'frame=%d' % f,
        })
    return data


def default_args(**over):
    a = dict(rtmin=0, rtmax=-1, mini=1, minmz=350, maxmz=1500, pasefmini=100,
             htol=8, itol=8, ignore_iso_calib=False, use_hill_calib=False,
             paseftol=0.05, nm=0, o='', iuse=0, hvf=1.3, ivf=5.0, minlh=2,
             pasefminlh=1, cmin=1, cmax=6, nprocs=1, dia=False, dia2=False,
             diahtol=25, diaminlh=1, diadynrange=1000, min_ms2_peaks=5, mgf='',
             debug=False, tof=False, tof_noise_factor=3.0, profile=False,
             write_hills=False, write_extra_details=False, md_correction='Orbi',
             combine_every=1, file='synthetic.mzML')
    a.update(over)
    return a


@pytest.fixture(scope='session')
def synth():
    return make_data()


@pytest.fixture
def run_pipeline(tmp_path, monkeypatch):
    """run_pipeline(data, fast=True, **args) -> (header, rows) of the features file."""
    from biosaur2 import main, utils, fastpath

    def _run(data, fast=True, **over):
        out = tmp_path / ('fast' if fast else 'fallback') / 'out.features.tsv'
        out.parent.mkdir(exist_ok=True)
        args = default_args(o=str(out), **over)
        monkeypatch.setattr(utils, 'process_mzml_stream', lambda *a, **k: None)
        monkeypatch.setattr(utils, 'process_mzml', lambda a: copy.deepcopy(data))
        if not fast:
            monkeypatch.setattr(fastpath, 'FAST_OK', False)
            monkeypatch.setattr(utils, 'centroid_pasef_scan_fast', None)
        main.process_file(args)
        monkeypatch.undo()
        lines = out.read_text().splitlines()
        return lines[0], lines[1:]
    return _run
