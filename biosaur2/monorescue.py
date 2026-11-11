"""
Monoisotopic rescue: move a feature's monoisotopic peak one 13C spacing down
when the evidence says it was placed on the M+1 isotope.

Features end up at M+1 when the hill at M does not form a valid isotope
cluster of its own, and in particular through the shifted-cosine branch of the
initial isotope search: above 1500 Da that branch asks whether a hill fits the
averagine pattern better as M+1 than as M, and when it does, it still records
the hill as monoisotopic. On PXD039875 about 8% of DDA precursors are matched
by a feature only at M+1. Quandenser ties MS2 spectra to MS1 features by m/z,
so those precursors lose their feature.

For each selected feature the rescue looks for a hill one isotope spacing
below its monoisotopic hill, at the same charge, that

  - no other feature has claimed, as monoisotope or isotope,
  - lies within the isotope m/z tolerance (and ion-mobility tolerance),
  - co-elutes with the monoisotopic hill (shared scans, profile cosine >= 0.6),
  - and extends the isotope pattern at least as well as it fit before: the
    averagine cosine over [candidate, current mono, isotopes...] must pass the
    same test the isotope search uses and be no lower than the feature's own.

The best such candidate becomes the monoisotopic hill and the old one becomes
isotope 1. Features without a candidate are left as they are, so the rescue
only moves features, it never removes or invents one.

It runs after feature selection, on whatever hills_dict the fast or the
original path produced, and touches only keys both provide.
"""
import logging

import numpy as np

from .cutils import get_and_calc_apex_intensity_and_scan, checking_cos_correlation_for_carbon

logger = logging.getLogger(__name__)

C13 = 1.00335
COS_RT_MIN = 0.6


def _profile_cosine(scans_1, ints_1, scans_2, ints_2):
    """Elution-profile cosine over shared scans, normalised by whole hills,
    as cutils.cos_correlation computes it."""
    d2 = dict(zip(scans_2, ints_2))
    top = 0.0
    for s, i in zip(scans_1, ints_1):
        j = d2.get(s)
        if j is not None:
            top += float(i) * float(j)
    if top == 0.0:
        return 0.0
    n1 = float(np.sqrt(np.sum(np.square(np.asarray(ints_1, dtype=np.float64)))))
    n2 = float(np.sqrt(np.sum(np.square(np.asarray(ints_2, dtype=np.float64)))))
    return top / (n1 * n2)


def rescue(hills_dict, features, used, averagine, isotopes_list, itol_ppm, paseftol):
    """Return (features, used) with monoisotopic peaks moved where justified.

    features   selected isotope clusters (ready_final)
    used       hill ids claimed by any feature (ready_set); updated in place
    averagine  the table built in process_features_iteration: mass bin ->
               (normalised isotope intensities, position of the maximum)
    """
    mz = np.asarray(hills_dict['hills_mz_median'], dtype=np.float64)
    order = np.argsort(mz, kind='stable')
    mz_sorted = mz[order]
    hill_ids = hills_dict['hills_idx_array_unique']
    im = hills_dict.get('hills_im_median') if paseftol > 0 else None
    scan_lists = hills_dict['hills_scan_lists']
    intensities = hills_dict['hills_intensity_array']

    moved = 0
    why = dict(no_hill=0, used=0, im=0, no_overlap=0, profile=0, pattern=0)
    for f in features:
        z = f['charge']
        mono_idx = f['monoisotope idx']
        m1 = f['hill_mz_1']
        target = m1 - C13 / z
        tol = itol_ppm * 1e-6 * target
        lo = np.searchsorted(mz_sorted, target - tol, 'left')
        hi = np.searchsorted(mz_sorted, target + tol, 'right')
        if hi <= lo:
            why['no_hill'] += 1
            continue

        scans_1 = scan_lists[mono_idx]
        ints_1 = intensities[mono_idx]
        first_1, last_1 = scans_1[0], scans_1[-1]
        theo_norm = averagine[int(100 * ((target * z) // 100))][0]
        exp_current = list(f['intensity_array_for_cos_corr'][1])
        cos_current = f['cos_cor_isotopes']

        best = None
        reasons = []
        for pos in order[lo:hi]:
            idx = int(pos)
            if hill_ids[idx] in used:
                reasons.append('used')
                continue
            if im is not None and abs(im[idx] - im[mono_idx]) > paseftol:
                reasons.append('im')
                continue
            scans_2 = scan_lists[idx]
            if scans_2[-1] < first_1 or scans_2[0] > last_1:
                reasons.append('no_overlap')
                continue
            cos_rt = _profile_cosine(scans_1, ints_1, scans_2, intensities[idx])
            if cos_rt < COS_RT_MIN:
                reasons.append('profile')
                continue
            hills_dict, apex_2, _ = get_and_calc_apex_intensity_and_scan(hills_dict, idx)
            theo = [apex_2 * theo_norm[k] / theo_norm[0] for k in isotopes_list]
            exp = [apex_2] + exp_current
            cos_ext, n_pass = checking_cos_correlation_for_carbon(theo, exp, 0.6)
            if n_pass < 2 or cos_ext < cos_current:
                reasons.append('pattern')
                continue
            if best is None or cos_ext > best[1]:
                best = (idx, cos_ext, n_pass, theo, exp, cos_rt)

        if best is None:
            # count each feature once, by the test its nearest candidate failed last
            for r in ('pattern', 'profile', 'no_overlap', 'im', 'used'):
                if r in reasons:
                    why[r] += 1
                    break
            continue

        idx, cos_ext, n_pass, theo, exp, cos_rt = best
        mz_new = float(mz[idx])
        expected = mz_new + C13 / z
        old_mono = {
            'isotope_number': 1,
            'isotope_hill_idx': f['monoisotope hill idx'],
            'isotope_idx': mono_idx,
            'cos_cor': cos_rt,
            'mass_diff_ppm': (m1 - expected) / expected * 1e6,
        }
        shifted = [dict(c, isotope_number=c['isotope_number'] + 1) for c in f['isotopes']]
        f['isotopes'] = ([old_mono] + shifted)[:n_pass - 1]
        f['monoisotope idx'] = idx
        f['monoisotope hill idx'] = hill_ids[idx]
        f['hill_mz_1'] = mz_new
        f['nIsotopes'] = n_pass
        f['cos_cor_isotopes'] = cos_ext
        f['intensity_array_for_cos_corr'] = [theo[:n_pass], exp[:n_pass]]
        if im is not None:
            f['im'] = im[idx]
        used.add(hill_ids[idx])
        moved += 1

    logger.info('Monoisotopic rescue moved %d of %d features one isotope down; '
                'not moved: %s', moved, len(features), why)
    return features, used
