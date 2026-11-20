"""Line-by-line transcriptions of the notebook's loops (plots, prints and DataFrames
removed). The package's vectorised code is tested against these."""

import numpy as np


def notebook_cfs(sig):
    """F1234cf_F4cfgradmeanx2_F4cfgrad_gtmean_... (cell 13)."""
    npts = len(sig)
    k = 100
    F1cf = np.zeros(npts)
    meany = np.zeros(npts)
    e4 = np.zeros(npts)
    e4mean = np.zeros(npts)
    e2 = np.zeros(npts)
    e2mean = np.zeros(npts)
    for i in range(k - 1, npts):
        meany[i] = sum(sig[i - k + 1 : i + 1]) / k
        e4[i] = (sig[i] - meany[i]) ** 4
        e4mean[i] = sum(e4[i - k + 1 : i + 1]) / k
        e2[i] = (sig[i] - meany[i]) ** 2
        e2mean[i] = sum(e2[i - k + 1 : i + 1]) / k
        F1cf[i] = (e4mean[i] / ((e2mean[i]) ** 2)) / 4
        F1cf[0 : (k - 1) * 2] = 0

    F2cf = np.zeros(npts)
    for x in range(1, len(F1cf)):
        dF = F1cf[x] - F1cf[x - 1]
        F2cf[x] = F2cf[x - 1] + dF if dF >= 0 else F2cf[x - 1]

    L = len(F2cf)
    AvgF2incr = (F2cf[-1] - F2cf[0]) / (L - 1)
    F3cf = np.zeros(npts)
    for y in range(L):
        F3cf[y] = F2cf[y] - (AvgF2incr * y)

    F4cf = np.zeros(npts)
    for m in range(0, len(F4cf) - 1):
        F4cf[m] = (F3cf[m] - F3cf[m + 1]) * -5 if F3cf[m] < F3cf[m + 1] else 0

    F4cfgrad = np.gradient(F4cf)
    for m in range(0, len(F4cf) - 1):
        if not F4cfgrad[m] > 0:
            F4cfgrad[m] = 0

    F4cfgradmeanx2 = np.ma.masked_where(F4cfgrad <= 0, F4cfgrad).mean() * 8
    F4cfgrad_gtmean = np.zeros(npts)
    for m in range(npts):
        F4cfgrad_gtmean[m] = F4cfgrad[m] if F4cfgrad[m] > F4cfgradmeanx2 else 0

    absValues = np.abs(sig)
    envwin = 8
    env_pos = np.zeros(npts)
    for i in range(envwin - 1, npts):
        env_pos[i] = (sum(absValues[i - envwin + 1 : i + 1]) / envwin) * 1.5

    return {
        "F1cf": F1cf, "F2cf": F2cf, "F3cf": F3cf, "F4cf": F4cf, "F4cfgrad": F4cfgrad,
        "threshold": float(F4cfgradmeanx2), "F4cfgrad_gtmean": F4cfgrad_gtmean, "env_pos": env_pos,
    }


def notebook_onset(F4cfgradmeanx2, F4cfgrad_gtmean, F1cf, F4cf):
    """sigOnset (cell 18), onset only."""
    F1cfMaxIndxnp = F1cf.argmax()
    idx = []
    for x in range(F1cfMaxIndxnp, F1cfMaxIndxnp - 7, -1):
        for j in range(1, 8):  # the seven elif branches
            if F4cfgrad_gtmean[x] > F4cfgradmeanx2 and F4cf[x - j] == 0:
                idx.append(x - j)
                break
    return int(np.unique(idx).max())


def notebook_subsequent_onsets(F1cf, F4cfgradmeanx2, F4cfgrad_gtmean, F4cf):
    """F4cfOnsets (cell 23), unique indices."""
    idx = []
    for x in range(F1cf.argmax(), len(F1cf)):
        if F4cfgrad_gtmean[x] > F4cfgradmeanx2 and F4cf[x - 1] == 0:
            idx.append(x - 1)
            continue
        for j in range(2, 6):
            if F4cfgrad_gtmean[x] > F4cfgradmeanx2 and F4cfgrad_gtmean[x - j] == 0 and F4cf[x - j] == 0:
                idx.append(x - j)
                break
    return np.unique(idx)


def notebook_preset_gain(Onset, SigSplineScale):
    """PreSetGainFunc (cell 22)."""
    GainFunc = SigSplineScale * 0.00
    OnsetSignalLength = 200
    OnsetplusT = Onset + OnsetSignalLength
    Position = np.arange(0, len(SigSplineScale), 1) - Onset
    GainFunc1 = np.power(1.01, Position)
    GainFunc3 = (1 - (np.exp((-Position + OnsetSignalLength) / 200))) + 1
    GainFunc[:Onset] = GainFunc1[:Onset]
    GainFunc[Onset:OnsetplusT] = 1.0
    GainFunc[OnsetplusT:] = GainFunc3[OnsetplusT:]
    return SigSplineScale * GainFunc


def notebook_trend(SigSplineScale, Onset):
    """SplineScale_Invert + SignalTrendData (cell 34): FuncMaxTrensFinalREInvert."""
    inv = 1 - abs(SigSplineScale)
    Sample = np.linspace(0, len(inv) - 1, len(inv))
    final = np.zeros(len(inv))
    xp, fp = np.empty((0, 0)), np.empty((0, 0))
    for x in range(Onset, len(inv), 20):
        xp = np.append(xp, min(inv[x : x + 20]))
        fp = np.append(fp, x + np.argmin(inv[x : x + 20]))
    minima = np.array(list(zip(fp, xp)))
    c = np.setdiff1d(Sample[Onset:], fp)
    trend = np.array(list(zip(c, np.interp(c, fp, xp))))
    combined = np.vstack((minima, trend)) if len(trend) else minima
    final[Onset:] = combined[np.argsort(combined[:, 0])][:, 1]
    return 1 - final


def notebook_gain_array(a, b, c, Onset, SigSplineScale):
    """GainFunctionArray (cell 34)."""
    Sample = np.linspace(0, len(SigSplineScale) - 1, len(SigSplineScale))
    G = np.empty(len(SigSplineScale))
    G.fill(1.000)
    for i in range(Onset, len(Sample)):
        G[i] = (a * np.exp(-b * (Sample[i] - Onset))) + c
    G = G / max(G)
    G[0:Onset] = 1.000
    return G, SigSplineScale / G


def notebook_xcorr_sum(a, others, Onset, include_self=False):
    """Xcorrs (cell 26) / AXcorrs (cell 30) for one target trace."""
    added = np.zeros(len(a))
    for b in ([a] if include_self else []) + list(others):
        Xcorr = np.correlate(a, b, mode="full")
        Xcorr_max = np.argmax(Xcorr) if abs(np.amax(Xcorr)) > abs(np.amin(Xcorr)) else np.argmin(Xcorr)
        Shift = int(Xcorr_max - Onset)
        XcorrShift = np.roll(Xcorr, -Shift)
        XcorrShift[len(b) + len(a) + 1 - Shift :] = 0
        added = added + XcorrShift[0 : (len(XcorrShift) - len(b) + 1)]
    return added


def notebook_impulse_corr(full, onset_a, impulse_sig):
    """ImpulseACorrs (cell 32) / ImpulseXcorrs (cell 48) inner loop, without mutating input."""
    FullSig = full.copy()
    FullSig[0:onset_a] = 0
    out = np.zeros(len(FullSig))
    for j in range(0, len(FullSig) - 60):
        out[j] = np.correlate(FullSig[j : j + 60], impulse_sig, mode="valid")[0]
    return out
