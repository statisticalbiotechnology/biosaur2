import ast
import copy

import numpy as np
import pytest

from conftest import make_data


def test_fast_equals_fallback(synth, run_pipeline):
    h1, fast = run_pipeline(synth, fast=True)
    h2, fb = run_pipeline(synth, fast=False)
    assert h1 == h2
    assert len(fast) > 20
    assert fast == fb


def test_fast_equals_fallback_options(synth, run_pipeline):
    opts = dict(md_correction='Icr', iuse=-1, write_extra_details=True, minlh=3)
    assert run_pipeline(synth, fast=True, **opts) == run_pipeline(synth, fast=False, **opts)


@pytest.mark.parametrize('fast', [True, False])
def test_no_hill_reaches_minlh(synth, run_pipeline, fast):
    # C3: used to raise UnboundLocalError in split_peaks_multi
    header, rows = run_pipeline(synth, fast=fast, minlh=1000)
    assert header.startswith('massCalib\t')
    assert rows == []


@pytest.mark.parametrize('fast', [True, False])
def test_no_isotope_clusters(run_pipeline, fast):
    # C3: used to raise IndexError on the empty cluster list
    data = make_data(n_peptides=20)
    header, rows = run_pipeline(data, fast=fast, cmin=6, cmax=6, itol=0.001)
    assert header.startswith('massCalib\t')
    assert rows == []


@pytest.mark.parametrize('fast', [True, False])
def test_isotopes_are_validated(run_pipeline, fast):
    # R3: every reported isotope passed the averagine cosine test; nIsotopes counts the mono too
    from biosaur2.cutils import checking_cos_correlation_for_carbon
    # 2nd 13C isotope 20x too intense: averagine explains only mono + first isotope
    data = make_data(n_peptides=80, distort=(2, 20.0), mass_range=(800, 1450))
    header, rows = run_pipeline(data, fast=fast, write_extra_details=True)
    cols = header.split('\t')
    i_n, i_iso, i_cc, i_m = (cols.index(c) for c in ('nIsotopes', 'isotopes', 'intensity_array_for_cos_corr', 'massCalib'))
    n_checked = 0
    for r in rows:
        f = r.split('\t')
        n = int(f[i_n])
        assert len(ast.literal_eval(f[i_iso])) == n - 1
        if float(f[i_m]) >= 1500:
            continue   # the shifted-cosine branch stores unshifted arrays (not changed)
        theor, exp = ast.literal_eval(f[i_cc])
        assert len(theor) == len(exp) == n
        cos, npass = checking_cos_correlation_for_carbon(list(theor), list(exp), 0.6)
        assert npass == n
        n_checked += 1
    assert n_checked > 10


def test_use_hill_calib(synth, run_pipeline):
    # C1: used to raise IndexError
    header, rows = run_pipeline(synth, fast=True, use_hill_calib=True)
    assert rows


def test_tof_dropped_spectrum_keeps_rt(run_pipeline):
    # C4: a spectrum removed by -tof must not shift the RTs of later scans
    # (process_tof estimates the noise from the first 25 spectra: insert after them)
    data = make_data(n_frames=60, n_peptides=80, noise_per_frame=800, seed=3)
    low = copy.deepcopy(data[26])
    low['intensity array'] = np.ones_like(low['intensity array'])   # below any noise threshold
    low['scanList'] = {'scan': [{'scan start time': 128.65}]}
    with_low = data[:26] + [low] + data[26:]
    _, a = run_pipeline(with_low, fast=True, tof=True)
    _, b = run_pipeline(data, fast=True, tof=True)
    assert a and a == b


def test_tof_single_noise_bin(run_pipeline):
    # C4: fewer than two calibrated noise bins used to raise IndexError
    data = make_data(n_frames=6, n_peptides=0, noise_per_frame=400)
    for z in data:
        sel = (z['m/z array'] >= 400) & (z['m/z array'] < 450)
        for k in ('m/z array', 'intensity array', 'mean inverse reduced ion mobility array'):
            z[k] = z[k][sel]
    header, rows = run_pipeline(data, fast=True, tof=True)
    assert header.startswith('massCalib\t')


def test_nprocs_falls_back_to_one(monkeypatch, tmp_path):
    # M1
    from biosaur2 import search, main
    seen = {}
    monkeypatch.setattr(main, 'process_file', lambda args: seen.update(args))
    monkeypatch.setattr('sys.argv', ['biosaur2', str(tmp_path / 'x.mzML'), '-nprocs', '4'])
    search.run()
    assert seen['nprocs'] == 1
