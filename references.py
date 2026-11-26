"""references.py - memuat gambar pose dari folder poses/ dan titik sensornya (pose_points.py)."""
import glob
import os

import cv2
import numpy as np

from config import *  # noqa: F401,F403
from scoring import idle_baseline
from ui import imread_safe, put

try:
    from pose_points import POSE_POINTS
except ImportError:
    POSE_POINTS = {}
    print("[!] File pose_points.py tidak ditemukan, titik pose dideteksi otomatis.")


def list_pose_files():
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"):
        files += glob.glob(os.path.join(POSE_DIR, ext))
    return sorted(set(files))


def points_from_config(fname):
    d = POSE_POINTS.get(fname)
    if not d:
        return None
    kp, conf = np.zeros((17, 2)), np.zeros(17)
    for nama, xy in d.items():
        if nama not in NAME2IDX:
            print(f"[!] {fname}: nama titik '{nama}' tidak dikenal, dilewati.")
            continue
        i = NAME2IDX[nama]
        kp[i] = xy
        conf[i] = 1.0
    if conf[5] == 0 or conf[6] == 0:
        print(f"[!] {fname}: bahu_kiri & bahu_kanan wajib diisi.")
        return None
    return kp, conf


def detect_person(model, img):
    res = model(img, verbose=False)[0]
    best, best_area = None, 0
    if res.keypoints is not None and len(res.keypoints) > 0:
        xy = res.keypoints.xy.cpu().numpy()
        cf = res.keypoints.conf.cpu().numpy()
        for kp, c in zip(xy, cf):
            if c[5] > 0.4 and c[6] > 0.4:
                v = kp[c > 0.3]
                area = np.ptp(v[:, 0]) * np.ptp(v[:, 1])
                if area > best_area:
                    best, best_area = (kp, c), area
    return best


def make_image_panel(img, name):
    card = np.full((PH, PW, 3), 245, np.uint8)
    h, w = img.shape[:2]
    f = min((PW - 20) / w, (PH - 70) / h)
    im = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))))
    x, y = (PW - im.shape[1]) // 2, 10
    card[y:y + im.shape[0], x:x + im.shape[1]] = im
    put(card, name[:22], (PW // 2, PH - 28), 0.9, (40, 40, 40), 2, True, False)
    return card


def load_references(model):
    refs = []
    for f in list_pose_files():
        img = imread_safe(f)
        if img is None:
            continue
        fname = os.path.basename(f)
        name = os.path.splitext(fname)[0]
        got, src = points_from_config(fname), "pose_points.py"
        if got is None:
            got, src = detect_person(model, img), "otomatis"
        if got is None:
            print(f"[!] {fname}: tidak ada di pose_points.py & orang tidak terdeteksi, dilewati.")
            continue
        base = idle_baseline(got[0], got[1])
        print(f"[ok] {fname}  (titik dari {src})  mirip-berdiri={base:.2f}")
        if base >= 0.5:
            print(f"     [!] {fname} mirip berdiri biasa, orang diam bisa dapat skor lumayan. Sebaiknya dibuang.")
        refs.append({"name": name, "kp": got[0], "conf": got[1], "base": base,
                     "panel": make_image_panel(img, name)})
    return refs