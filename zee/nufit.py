"""NuFIT 6.1 chi^2 from the released projection tables (v61.release-TB{yes,off}-{NO,IO}.txt.xz).

chi2 = 3D(s23, Dm3l, delta) + [2D(s13,s23) - 1D(s23)] + 2D(s12, log10 Dm21)
       + [2D(s13,s12) - 1D(s13) - 1D(s12)]

APPROXIMATION: the file has only projections. The 3D table carries the (theta23, Dm3l,
delta) correlations, the 2D tables add the theta13-theta23, theta12-Dm21 and
theta12-theta13 correlations by inclusion-exclusion. Remaining dropped: theta13/theta12
vs Dm3l and delta, Dm21 vs the rest. Not a profile of the full likelihood.
The tables are Dchi2 relative to the GLOBAL (NO) minimum, so each IO projection already
contains the NO-IO offset (5.91 for TByes). It is subtracted from the 1D terms here so it
is counted once: chi2 is relative to the global minimum for both orderings.
"""
import lzma
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_CACHE = os.path.join(os.path.dirname(__file__), "_nufit_cache")


def _parse(path):
    secs, cur, name = {}, [], None
    with lzma.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                if name is not None and cur:
                    secs[name] = np.array(cur, float)
                name, cur = line.split()[1], []
            elif line.strip() and not line.startswith("###"):
                cur.append(line.split())
    if name is not None and cur:
        secs[name] = np.array(cur, float)
    return secs


def build_cache(path, tag):
    os.makedirs(_CACHE, exist_ok=True)
    secs = _parse(path)
    out = {}
    for key in ("T12", "T13", "T23", "DMA", "DMS", "DCP"):
        out[key] = secs[key]
    for key in ("T13/T23", "T12/DMS", "T13/T12"):
        d2 = secs[key]
        a0, a1 = np.unique(d2[:, 0]), np.unique(d2[:, 1])
        out["g2_" + key] = d2[:, 2].reshape(len(a0), len(a1))
        out["a0_" + key], out["a1_" + key] = a0, a1
    d = secs["T23/DMA/DCP"]
    ax = [np.unique(d[:, k]) for k in range(3)]
    grid = d[:, 3].reshape(len(ax[0]), len(ax[1]), len(ax[2]))
    out["ax0"], out["ax1"], out["ax2"], out["grid3d"] = ax[0], ax[1], ax[2], grid
    np.savez_compressed(os.path.join(_CACHE, tag + ".npz"), **out)


class NuFit:
    def __init__(self, ordering="NO", tb=True, source_dir=None):
        tag = f"TB{'yes' if tb else 'off'}-{ordering}"
        f = os.path.join(_CACHE, tag + ".npz")
        if not os.path.exists(f):
            import glob
            hits = glob.glob(os.path.join(source_dir or "", f"*v61.release-{tag}.txt.xz"))
            if not hits:
                raise FileNotFoundError(f"no cache and no source for {tag}")
            build_cache(hits[0], tag)
        z = np.load(f)
        self.ordering = ordering
        self._p3 = RegularGridInterpolator((z["ax0"], z["ax1"], z["ax2"]), z["grid3d"],
                                           bounds_error=False, fill_value=None)
        self.dmin = float(z["grid3d"].min())  # ordering offset w.r.t. global minimum
        self._lo = np.array([z["ax0"][0], z["ax1"][0], z["ax2"][0]])
        self._hi = np.array([z["ax0"][-1], z["ax1"][-1], z["ax2"][-1]])
        self._1d = {k: (z[k][:, 0], z[k][:, 1]) for k in ("T12", "T13", "T23", "DMS")}
        self._2d = {k: RegularGridInterpolator((z["a0_" + k], z["a1_" + k]), z["g2_" + k],
                                               bounds_error=False, fill_value=None)
                    for k in ("T13/T23", "T12/DMS", "T13/T12")}
        self._2lim = {k: (z["a0_" + k][[0, -1]], z["a1_" + k][[0, -1]]) for k in self._2d}
        self.delta_range = (z["ax2"][0], z["ax2"][-1])

    def _chi1(self, key, x):
        xs, ys = self._1d[key]
        return float(np.interp(x, xs, ys - self.dmin, left=ys[0] + 1e3, right=ys[-1] + 1e3))

    def chi2(self, s12sq, s13sq, s23sq, dm21_eV2, dm3l_eV2, delta_deg):
        """Dchi2 (approximate, relative to the table minimum)."""
        lo, hi = self.delta_range
        d = ((delta_deg - lo) % 360.0) + lo if hi - lo >= 359.9 else delta_deg
        p = np.array([s23sq, dm3l_eV2 * 1e3, d])
        out = (p < self._lo).any() or (p > self._hi).any()
        c3 = float(self._p3(p)[0]) if not out else 1e4
        # inclusion-exclusion over pair tables (each term relative to the ordering minimum)
        c = c3 - self.dmin
        c += self._pair("T13/T23", s13sq, s23sq) - self._chi1("T23", s23sq)
        c += self._pair("T12/DMS", s12sq, np.log10(dm21_eV2))
        c += (self._pair("T13/T12", s13sq, s12sq) - self._chi1("T13", s13sq)
              - self._chi1("T12", s12sq))
        return c + self.dmin

    def _pair(self, key, x, y):
        (x0, x1), (y0, y1) = self._2lim[key]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return 1e4
        return float(self._2d[key]([[x, y]])[0]) - self.dmin
