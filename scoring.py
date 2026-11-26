"""scoring.py - perhitungan skor kemiripan pose (NumPy) dan cek badan terlihat."""
import math

import numpy as np

from config import *  # noqa: F401,F403

# posisi berdiri netral (lebar bahu = 1, tengah bahu = titik 0, sumbu y ke bawah)
NEUTRAL = np.zeros((17, 2))
NEUTRAL[5], NEUTRAL[6] = (0.5, 0), (-0.5, 0)
NEUTRAL[7], NEUTRAL[8] = (0.55, 0.85), (-0.55, 0.85)
NEUTRAL[9], NEUTRAL[10] = (0.55, 1.55), (-0.55, 1.55)
NEUTRAL[11], NEUTRAL[12] = (0.37, 1.3), (-0.37, 1.3)
NEUTRAL[13], NEUTRAL[14] = (0.37, 2.5), (-0.37, 2.5)
NEUTRAL[15], NEUTRAL[16] = (0.37, 3.7), (-0.37, 3.7)

SCORED_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]


def normalize(kp, conf, thr=0.3):
    valid = np.asarray(conf) > thr
    if not (valid[5] and valid[6]):
        return None, valid
    kp = np.asarray(kp, dtype=float)
    center = (kp[5] + kp[6]) / 2
    scale = np.linalg.norm(kp[5] - kp[6])
    if scale < 1e-6:
        return None, valid
    return (kp - center) / scale, valid


def _adiff(v1, v2):
    """Selisih arah dua vektor dalam derajat (0-180)."""
    a = math.degrees(math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0]))
    return abs((a + 180) % 360 - 180)


def _soft(d, forgive, tol):
    """1.0 kalau selisih <= forgive, lalu turun halus."""
    return 1.0 if d <= forgive else math.exp(-((d - forgive) / tol) ** 2)


def _raw_sim(ref_kp, ref_conf, kp, conf):
    """Skor mentah 0..1."""
    r, rv = normalize(ref_kp, ref_conf)
    p, pv = normalize(kp, conf)
    if r is None or p is None:
        return 0.0

    # 1) arah anggota badan
    num = den = 0.0
    for a, b in SCORED_LIMBS:
        if not (rv[a] and rv[b]):
            continue
        vr = r[b] - r[a]
        if np.linalg.norm(vr) < 1e-6:
            continue
        w = 0.1 + min(1.0, _adiff(vr, NEUTRAL[b] - NEUTRAL[a]) / 60.0)
        s = 0.0
        if pv[a] and pv[b]:
            vp = p[b] - p[a]
            if np.linalg.norm(vp) > 1e-6:
                s = _soft(_adiff(vr, vp), ANGLE_FORGIVE, ANGLE_TOL)
        num += w * s
        den += w
    ang_score = num / den if den else 0.0

    # 2) posisi siku & tangan
    pn = pd = 0.0
    for i in (7, 8, 9, 10):
        if not rv[i]:
            continue
        w = 0.1 + min(1.0, float(np.linalg.norm(r[i] - NEUTRAL[i])) / 0.8)
        s = _soft(float(np.linalg.norm(r[i] - p[i])), POS_FORGIVE, SIGMA) if pv[i] else 0.0
        pn += w * s
        pd += w
    pos_score = pn / pd if pd else 0.0

    # 3) kemiringan bahu (hanya kalau di gambar memang miring)
    def _tilt(q):
        return math.degrees(math.atan2(q[5][1] - q[6][1], q[5][0] - q[6][0]))
    rt = _tilt(r)
    wt = min(1.0, abs(rt) / 20.0)
    d = abs((rt - _tilt(p) + 180) % 360 - 180)
    tilt_score = _soft(d, TILT_FORGIVE, 25.0)

    total = (0.65 * ang_score + 0.35 * pos_score + wt * tilt_score) / (1.0 + wt)
    vis = float(np.mean([pv[i] for i in range(7, 17)]))
    return total * (0.85 + 0.15 * vis)


def idle_baseline(ref_kp, ref_conf):
    """Skor mentah orang yang berdiri diam terhadap pose ini."""
    return min(_raw_sim(ref_kp, ref_conf, NEUTRAL, np.ones(17)), BASE_CAP)


def similarity(ref_kp, ref_conf, kp, conf, base=0.0):
    """Skor 0-100. Berdiri diam = 0, meniru dengan benar = 100."""
    raw = _raw_sim(ref_kp, ref_conf, kp, conf)
    s = (raw - base) / max(1e-6, 1.0 - base)
    return float(100 * min(1.0, max(0.0, s)))


def body_visible(p):
    kp, c = p["kp"], p["conf"]
    for i in CORE:
        x, y = kp[i]
        if c[i] < CONF_MIN or not (MARGIN <= x <= W - MARGIN and MARGIN <= y <= H - MARGIN):
            return False
    return c[0] > 0.3 and kp[0][1] >= MARGIN
