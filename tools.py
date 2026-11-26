"""tools.py - alat bantu (bukan bagian game):
   python ver2.py cari poses/gambar.jpg   -> klik gambar untuk melihat koordinat
   python ver2.py lihat                   -> pratinjau skeleton dari pose_points.py
"""
import os

import cv2
import numpy as np

from references import list_pose_files, points_from_config
from ui import imread_safe, put


def coord_helper(path):
    img = imread_safe(path)
    if img is None:
        raise SystemExit(f"Gambar tidak ditemukan: {path}")
    ih, iw = img.shape[:2]
    sc = min(1.0, 900 / ih, 1400 / iw)
    base = cv2.resize(img, (int(iw * sc), int(ih * sc)))
    marks, cur = [], [(0, 0)]

    def cb(ev, x, y, fl, pr):
        cur[0] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            ox, oy = int(x / sc), int(y / sc)
            marks.append((x, y, ox, oy))
            print(f"({ox}, {oy})")

    print(f"Ukuran gambar: {iw} x {ih}. Klik tiap bagian tubuh, Z = undo, ESC = selesai.")
    cv2.namedWindow("Cari Koordinat")
    cv2.setMouseCallback("Cari Koordinat", cb)
    while True:
        view = base.copy()
        for n, (x, y, ox, oy) in enumerate(marks):
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, f"{n + 1}: ({ox},{oy})", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
        mx, my = cur[0]
        cv2.putText(view, f"kursor: ({int(mx / sc)}, {int(my / sc)})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Cari Koordinat", view)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        if k in (ord("z"), ord("Z")) and marks:
            marks.pop()
    cv2.destroyAllWindows()


def preview_poses():
    """Foto di kiri, skeleton dari pose_points.py di kanan. B = sebelumnya, ESC = keluar."""
    EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    os.makedirs("preview", exist_ok=True)
    items = []
    for f in list_pose_files():
        fname = os.path.basename(f)
        got = points_from_config(fname)
        img = imread_safe(f)
        if got is None or img is None:
            print(f"[lewati] {fname}: tidak ada di pose_points.py")
            continue
        items.append((fname, img, got))
    if not items:
        raise SystemExit("Tidak ada pose untuk ditampilkan.")

    CH = 640
    i = 0
    while True:
        fname, img, (kp, conf) = items[i]
        h, w = img.shape[:2]
        left = cv2.resize(img, (int(w * CH / h), CH))
        right = np.full((CH, 560, 3), 245, np.uint8)
        c0 = (kp[5] + kp[6]) / 2
        sc = 100 / max(np.linalg.norm(kp[5] - kp[6]), 1e-6)
        pt = lambda j: (int(280 + (kp[j][0] - c0[0]) * sc), int(140 + (kp[j][1] - c0[1]) * sc))
        for a, b in EDGES:
            if conf[a] and conf[b]:
                cv2.line(right, pt(a), pt(b), (90, 90, 90), 6, cv2.LINE_AA)

        u = kp[5] - kp[6]
        u = u / max(np.linalg.norm(u), 1e-6)
        n = np.array([u[1], -u[0]])
        mid = np.array([280.0, 140.0])
        neck = (mid + n * 38).astype(int)
        head = (mid + n * 72).astype(int)
        cv2.line(right, (280, 140), tuple(neck), (90, 90, 90), 6, cv2.LINE_AA)
        cv2.circle(right, tuple(head), 35, (90, 90, 90), 6, cv2.LINE_AA)

        for j in range(5, 17):
            if conf[j]:
                col = (220, 120, 0) if j % 2 == 0 else (0, 140, 255)
                cv2.circle(right, pt(j), 11, col, -1, cv2.LINE_AA)
        put(right, "kiri gambar", (90, 620), 0.6, (220, 120, 0), 1, True, False)
        put(right, "kanan gambar", (470, 620), 0.6, (0, 140, 255), 1, True, False)
        view = np.hstack([left, right])
        put(view, f"{i + 1}/{len(items)}  {fname}", (view.shape[1] // 2, 25), 0.7, (0, 0, 0), 2, True, False)
        cv2.imwrite(os.path.join("preview", os.path.splitext(fname)[0] + ".jpg"), view)
        cv2.imshow("Pratinjau pose (B = sebelumnya, ESC = keluar)", view)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
        i = (i - 1) % len(items) if k in (ord("b"), ord("B")) else (i + 1) % len(items)
    cv2.destroyAllWindows()
    print("Gambar pratinjau juga tersimpan di folder 'preview'.")