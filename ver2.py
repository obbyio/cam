"""
POSE BATTLE - Game adu gerakan 2 pemain dengan webcam
Layout : panel KIRI = Player 1, TENGAH = gambar pose, panel KANAN = Player 2
Titik sensor tiap gambar ada di file terpisah: pose_points.py
Suara di folder sounds/ (lihat daftar nama file di bagian PENGATURAN SUARA)
Cari koordinat gambar:  python ver2.py cari poses/pose1.jpg
Kontrol: PLAY (klik / ENTER), SPASI = lanjut ronde, F = fullscreen, ESC = menu, Q = keluar
Instal : pip install ultralytics opencv-python numpy pygame
"""
import datetime
import glob
import math
import os
import random
import sys
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pygame
except ImportError:
    pygame = None
    print("[!] pygame belum terpasang (pip install pygame), game jalan tanpa suara.")

try:
    from pose_points import POSE_POINTS
except ImportError:
    POSE_POINTS = {}
    print("[!] File pose_points.py tidak ditemukan, titik pose dideteksi otomatis.")

# ======================= PENGATURAN =======================
W, H = 1280, 720
ROUNDS = 6
POSE_TIME = 5
COUNTDOWN_TIME = 3
RESULT_TIME = 5
REC_FPS = 20
MODEL_NAME = "yolov8n-pose.pt"
POSE_DIR, REC_DIR, SNAP_DIR = "poses", "recordings", "snapshots"
SIGMA = 0.5               # makin kecil = penilaian makin ketat
PW, PH = 320, 400         # ukuran gambar contoh di tengah
NAME_MAX = 12
WIN = "Pose Battle"
SENSOR_ON = True          # True = sensor aktif | False = mode TEST (skor acak)
FONT = cv2.FONT_HERSHEY_DUPLEX
P_COLORS = [(255, 200, 0), (180, 90, 255)]
GOLD = (0, 215, 255)
ANGLE_TOL = 40.0  

TILT_FORGIVE = 8.0        # selisih kemiringan bahu yang dimaafkan (derajat)
SCORE_WINDOW = 1.0        # skor diambil dari N detik terakhir (median)

# ======================= PENGATURAN SUARA =======================
SOUND_ON = True           # False = matikan semua suara
SOUND_DIR = "sounds"
MUSIC_VOL = 0.5           # volume musik latar (0.0 - 1.0)
SFX_VOL = 0.9             # volume efek suara (0.0 - 1.0)
# nama file (tanpa ekstensi; boleh .wav / .mp3 / .ogg)
SFX_FILES = {
    "klik": "klik",
    "hitung": "hitung",
    "mulai": "mulai",
    "pose5": "pose5detik",
    "jepret": "jepret",
}
MUSIC_FILES = {
    "menu": "musik_menu",
    "main": "musik_main",
    "leaderboard": "musik_leaderboard",
}
# musik yang diputar di tiap layar (None = tanpa musik)
MUSIC_STATE = {
    "MENU": "menu", "NAMES": "menu", "WAIT": "menu",
    "COUNTDOWN": "main", "POSE": "main", "RESULT": "main",
    "FINAL": "leaderboard", "PLAYBACK": None,
}

# Layout panel pemain
SW = 445
PT, PB = 70, 50
PAN_H = H - PT - PB
CROP_W = int(H * SW / PAN_H)
PANELS = [(5, PT), (W - 5 - SW, PT)]

NAME2IDX = {
    "hidung": 0,
    "bahu_kiri": 6, "bahu_kanan": 5,
    "siku_kiri": 8, "siku_kanan": 7,
    "tangan_kiri": 10, "tangan_kanan": 9,
    "pinggul_kiri": 12, "pinggul_kanan": 11,
    "lutut_kiri": 14, "lutut_kanan": 13,
    "kaki_kiri": 16, "kaki_kanan": 15,
}

CORE = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]   # bahu, siku, pinggul, lutut, pergelangan kaki
CONF_MIN = 0.5
MARGIN = 8

BODY = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14),
         (14, 16), (5, 11), (6, 12), (5, 6), (11, 12)]


# ======================= SUARA =======================
def find_sound(base):
    for ext in (".wav", ".mp3", ".ogg"):
        p = os.path.join(SOUND_DIR, base + ext)
        if os.path.exists(p):
            return p
    return None


class Audio:
    """Efek suara + musik latar. File yang tidak ada dilewati tanpa error."""

    def __init__(self):
        self.ok = False
        self.sfx, self.music, self.cur = {}, {}, "?"
        if not SOUND_ON or pygame is None:
            return
        try:
            pygame.mixer.init()
        except Exception as e:
            print("[!] Audio tidak bisa dijalankan:", e)
            return
        self.ok = True
        os.makedirs(SOUND_DIR, exist_ok=True)
        for key, base in SFX_FILES.items():
            p = find_sound(base)
            if p:
                try:
                    s = pygame.mixer.Sound(p)
                    s.set_volume(SFX_VOL)
                    self.sfx[key] = s
                    print(f"[suara ok] {key}: {p}")
                except Exception as e:
                    print(f"[!] Gagal memuat {p}: {e}")
            else:
                print(f"[suara] tidak ada: {SOUND_DIR}/{base}.wav/.mp3/.ogg")
        for key, base in MUSIC_FILES.items():
            p = find_sound(base)
            if p:
                self.music[key] = p
                print(f"[musik ok] {key}: {p}")
            else:
                print(f"[musik] tidak ada: {SOUND_DIR}/{base}.wav/.mp3/.ogg")

    def play(self, key):
        if self.ok and key in self.sfx:
            self.sfx[key].play()

    def stop_sfx(self):
        if self.ok:
            pygame.mixer.stop()

    def set_music(self, key):
        """Ganti musik latar (tidak mengulang dari awal kalau musiknya sama)."""
        if not self.ok or key == self.cur:
            return
        self.cur = key
        try:
            pygame.mixer.music.stop()
            if key and key in self.music:
                pygame.mixer.music.load(self.music[key])
                pygame.mixer.music.set_volume(MUSIC_VOL)
                pygame.mixer.music.play(-1)
        except Exception as e:
            print("[!] Musik gagal diputar:", e)

    def close(self):
        if self.ok:
            pygame.mixer.quit()


# ======================= PERHITUNGAN KEMIRIPAN (NUMPY) =======================
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


# posisi berdiri netral (lebar bahu = 1, tengah bahu = titik 0, sumbu y ke bawah)
NEUTRAL = np.zeros((17, 2))
NEUTRAL[5], NEUTRAL[6] = (0.5, 0), (-0.5, 0)
NEUTRAL[7], NEUTRAL[8] = (0.55, 0.85), (-0.55, 0.85)
NEUTRAL[9], NEUTRAL[10] = (0.55, 1.55), (-0.55, 1.55)
NEUTRAL[11], NEUTRAL[12] = (0.37, 1.3), (-0.37, 1.3)
NEUTRAL[13], NEUTRAL[14] = (0.37, 2.5), (-0.37, 2.5)
NEUTRAL[15], NEUTRAL[16] = (0.37, 3.7), (-0.37, 3.7)

SCORED_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]


def _adiff(v1, v2):
    """Selisih arah dua vektor dalam derajat (0-180)."""
    a = math.degrees(math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0]))
    return abs((a + 180) % 360 - 180)


def similarity(ref_kp, ref_conf, kp, conf):
    """Skor 0-100. Bagian tubuh yang di gambar BEDA dari posisi berdiri biasa
    bobotnya besar; bagian yang cuma berdiri lurus bobotnya kecil.
    Jadi orang yang diam saja skornya rendah."""
    r, rv = normalize(ref_kp, ref_conf)
    p, pv = normalize(kp, conf)
    if r is None or p is None:
        return 0.0

    # 1) arah anggota badan (proporsi tubuh tidak berpengaruh)
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
                s = math.exp(-(_adiff(vr, vp) / ANGLE_TOL) ** 2)
        num += w * s
        den += w
    ang_score = num / den if den else 0.0

    # 2) posisi siku & tangan relatif terhadap bahu
    pn = pd = 0.0
    for i in (7, 8, 9, 10):
        if not rv[i]:
            continue
        w = 0.1 + min(1.0, float(np.linalg.norm(r[i] - NEUTRAL[i])) / 0.8)
        s = math.exp(-(float(np.linalg.norm(r[i] - p[i])) / SIGMA) ** 2) if pv[i] else 0.0
        pn += w * s
        pd += w
    pos_score = pn / pd if pd else 0.0

    # 3) kemiringan bahu: hanya dihitung kalau di gambar memang miring
    def _tilt(q):
        return math.degrees(math.atan2(q[5][1] - q[6][1], q[5][0] - q[6][0]))
    rt = _tilt(r)
    wt = min(1.0, abs(rt) / 20.0)
    d = max(0.0, abs((rt - _tilt(p) + 180) % 360 - 180) - TILT_FORGIVE)
    tilt_score = math.exp(-(d / 25.0) ** 2)

    total = (0.65 * ang_score + 0.35 * pos_score + wt * tilt_score) / (1.0 + wt)

    # titik yang tidak terlihat mengurangi skor (tidak bisa sempurna kalau tubuh terpotong)
    vis = float(np.mean([pv[i] for i in range(7, 17)]))
    return float(100 * total * (0.85 + 0.15 * vis))

def body_visible(p):
    kp, c = p["kp"], p["conf"]
    for i in CORE:
        x, y = kp[i]
        if c[i] < CONF_MIN or not (MARGIN <= x <= W - MARGIN and MARGIN <= y <= H - MARGIN):
            return False
    return c[0] > 0.3 and kp[0][1] >= MARGIN


# ======================= UTILITAS TAMPILAN =======================
def imread_safe(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def put(img, s, pos, scale=1.0, color=(255, 255, 255), thick=2, center=False, outline=True):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = pos
    if center:
        x, y = x - tw // 2, y + th // 2
    if outline:
        cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def shade(img, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.5):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full(roi.shape, color, np.uint8), alpha, 0)


def fit_frame(img):
    h, w = img.shape[:2]
    target = W / H
    if w / h > target:
        nw = int(h * target)
        x = (w - nw) // 2
        img = img[:, x:x + nw]
    elif w / h < target:
        nh = int(w / target)
        y = (h - nh) // 2
        img = img[y:y + nh]
    return cv2.resize(img, (W, H))


class Button:
    def __init__(self, label, x, y, w, h, color):
        self.label, self.x, self.y, self.w, self.h, self.color = label, x, y, w, h, color

    def hit(self, p):
        return p is not None and self.x <= p[0] <= self.x + self.w and self.y <= p[1] <= self.y + self.h

    def draw(self, img, mouse):
        c = tuple(min(255, int(v * 1.25)) for v in self.color) if self.hit(mouse) else self.color
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), c, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 3)
        put(img, self.label, (self.x + self.w // 2, self.y + self.h // 2), 1.0, (255, 255, 255), 2, True)


# ======================= ALAT BANTU: CARI KOORDINAT =======================
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


# ======================= GAMBAR CONTOH (dari folder poses/) =======================
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
        print(f"[ok] {fname}  (titik dari {src})")
        refs.append({"name": name, "kp": got[0], "conf": got[1],
                     "panel": make_image_panel(img, name)})
    return refs

def preview_poses():
    """python ver2.py lihat -> foto di kiri, skeleton dari pose_points.py di kanan.
    Tombol apa saja = berikutnya, B = sebelumnya, ESC = keluar."""
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
        # skeleton: pusat = tengah bahu, skala = lebar bahu
        c0 = (kp[5] + kp[6]) / 2
        sc = 100 / max(np.linalg.norm(kp[5] - kp[6]), 1e-6)
        pt = lambda j: (int(280 + (kp[j][0] - c0[0]) * sc), int(140 + (kp[j][1] - c0[1]) * sc))
        for a, b in EDGES:
            if conf[a] and conf[b]:
                cv2.line(right, pt(a), pt(b), (90, 90, 90), 6, cv2.LINE_AA)
                
        # kepala: tegak lurus garis bahu, jadi ikut miring
        u = kp[5] - kp[6]
        u = u / max(np.linalg.norm(u), 1e-6)
        n = np.array([u[1], -u[0]])                       # arah "atas" badan
        mid = np.array([280.0, 140.0])
        neck = (mid + n * 38).astype(int)
        head = (mid + n * 72).astype(int)
        cv2.line(right, (280, 140), tuple(neck), (90, 90, 90), 6, cv2.LINE_AA)
        cv2.circle(right, tuple(head), 35, (90, 90, 90), 6, cv2.LINE_AA)

        for j in range(5, 17):
            if conf[j]:
                col = (220, 120, 0) if j % 2 == 0 else (0, 140, 255)   # biru = kiri gambar, oranye = kanan gambar
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

# ======================= GAME =======================
class PoseBattle:
    def __init__(self):
        print("Memuat model...")
        self.model = YOLO(MODEL_NAME)
        self.model(np.zeros((640, 640, 3), np.uint8), verbose=False)
        os.makedirs(POSE_DIR, exist_ok=True)
        self.refs = load_references(self.model)
        self.nrounds = min(ROUNDS, len(self.refs))
        print("MODE SENSOR:", "AKTIF" if SENSOR_ON else "TEST (sensor mati)")
        self.audio = Audio()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise SystemExit("Webcam tidak ditemukan.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.mouse, self.click = (0, 0), None
        self.fullscreen = False
        self.quit = False
        self.state = "MENU"
        self.writer = None
        self.rec_path = None
        self.snap_dir = None
        self.pb = None
        self.pb_frame = None
        self.pb_last = 0
        self.partial = [False, False]
        self.raw = [None, None]
        self.buf = [[], []]
        self.last_cd = None           # angka hitung mundur terakhir yang sudah dibunyikan
        self.saved = False
        self.names = ["", ""]
        self.active = 0
        self.cam_x = [W * 0.25, W * 0.75]

        self.btn_play = Button("PLAY", W // 2 - 160, 400, 320, 80, (60, 170, 60))
        self.btn_quit = Button("KELUAR", W // 2 - 160, 500, 320, 70, (60, 60, 170))
        self.btn_cancel = Button("BATAL", 40, 630, 200, 60, (60, 60, 170))
        self.btn_replay = Button("REKAMAN", 55, 620, 270, 70, (170, 110, 40))
        self.btn_again = Button("MAIN LAGI", 355, 620, 270, 70, (60, 170, 60))
        self.btn_tomenu = Button("MENU", 655, 620, 270, 70, (170, 110, 40))
        self.btn_exit = Button("KELUAR", 955, 620, 270, 70, (60, 60, 170))
        self.btn_back = Button("KEMBALI", W - 240, H - 90, 200, 60, (60, 60, 170))
        self.btn_start = Button("MULAI", W // 2 + 20, 560, 280, 80, (60, 170, 60))
        self.btn_menu = Button("KEMBALI", W // 2 - 300, 560, 280, 80, (60, 60, 170))
        self.field_rects = [(W // 2 - 300, 220, 600, 70), (W // 2 - 300, 370, 600, 70)]
        self.reset_game()

    # ---------- input ----------
    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def click_on_button(self):
        """True jika klik terakhir mengenai tombol di layar yang sedang tampil."""
        bs = {
            "MENU": [self.btn_play, self.btn_quit],
            "NAMES": [self.btn_menu, self.btn_start],
            "WAIT": [self.btn_cancel],
            "FINAL": [self.btn_replay, self.btn_again, self.btn_tomenu, self.btn_exit],
            "PLAYBACK": [self.btn_back],
        }.get(self.state, [])
        return any(b.hit(self.click) for b in bs)

    def pname(self, i):
        n = self.names[i].strip()
        return n if n else f"Player {i + 1}"

    # ---------- deteksi pemain ----------
    def detect(self, frame):
        """Kembalikan [P1, P2]; P1 = orang di kiri kamera, P2 = orang di kanan."""
        if not SENSOR_ON:                      # MODE TEST
            self.partial = [False, False]
            fake = []
            for i in range(2):
                cx = W * (0.25 if i == 0 else 0.75)
                kp = np.zeros((17, 2))
                kp[0] = (cx, 260)
                kp[5], kp[6] = (cx + 60, 380), (cx - 60, 380)
                kp[7], kp[8] = (cx + 100, 470), (cx - 100, 470)
                kp[9], kp[10] = (cx + 120, 560), (cx - 120, 560)
                fake.append({"kp": kp, "conf": np.ones(17), "area": 1, "cx": cx})
            self.raw = fake
            return fake

        slots = [None, None]
        self.partial = [False, False]
        self.raw = [None, None]
        res = self.model(frame, imgsz=640, conf=0.4, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return slots
        xy = res.keypoints.xy.cpu().numpy()
        cf = res.keypoints.conf
        cf = cf.cpu().numpy() if cf is not None else np.ones(xy.shape[:2])
        people = []
        for kp, c in zip(xy, cf):
            c = c.copy()
            out = (kp[:, 0] < MARGIN) | (kp[:, 0] > W - MARGIN) | (kp[:, 1] < MARGIN) | (kp[:, 1] > H - MARGIN)
            c[out] = 0                          # titik di luar layar dianggap tidak terlihat
            if c[5] > 0.4 and c[6] > 0.4:
                v = kp[c > 0.3]
                people.append({"kp": kp, "conf": c,
                               "area": np.ptp(v[:, 0]) * np.ptp(v[:, 1]),
                               "cx": v[:, 0].mean()})
        people.sort(key=lambda p: -p["area"])
        people = sorted(people[:2], key=lambda p: p["cx"])
        if len(people) == 2:
            slots = people
        elif len(people) == 1:
            slots[0 if people[0]["cx"] < W / 2 else 1] = people[0]

        self.raw = list(slots)
        if self.state in ("WAIT", "COUNTDOWN"):
            for i, p in enumerate(slots):
                if p is not None and not body_visible(p):
                    self.partial[i] = True
                    slots[i] = None
        return slots

    # ---------- tampilan 3 zona: kiri | gambar | kanan ----------
    def compose(self, frame, players, header=False):
        screen = np.full((H, W, 3), (25, 22, 22), np.uint8)
        sc = PAN_H / H
        for i in range(2):
            p = players[i]
            if p is not None:
                self.cam_x[i] += 0.25 * (p["cx"] - self.cam_x[i])
            x0 = int(np.clip(self.cam_x[i] - CROP_W / 2, 0, W - CROP_W))
            crop = cv2.resize(frame[:, x0:x0 + CROP_W], (SW, PAN_H))
            px, py = PANELS[i]
            screen[py:py + PAN_H, px:px + SW] = crop
            cv2.rectangle(screen, (px, py), (px + SW, py + PAN_H), P_COLORS[i], 5)

            if header:
                put(screen, self.pname(i), (px + SW // 2, 35), 1.0, P_COLORS[i], 2, True)

            if p is not None:
                kp, c = p["kp"], p["conf"]
                v = kp[c > 0.3]
                sw_ = np.linalg.norm(kp[5] - kp[6])
                hx = int(v[:, 0].mean())
                hy = int(kp[0][1] - 0.9 * sw_) if c[0] > 0.3 else int(v[:, 1].min() - 45)
                lx = int(np.clip(px + (hx - x0) * sc, px + 70, px + SW - 70))
                ly = int(max(py + 25, py + hy * sc))
                put(screen, self.pname(i), (lx, ly), 0.8, P_COLORS[i], 2, True)
        return screen

    # ---------- siluet panduan (tahap cek sensor) ----------
    def _vis(self, p, i):
        x, y = p["kp"][i]
        return p["conf"][i] > CONF_MIN and MARGIN <= x <= W - MARGIN and MARGIN <= y <= H - MARGIN

    def _pair(self, p, a, b, cx):
        if p is None:
            return False, False
        va, vb = self._vis(p, a), self._vis(p, b)
        if va and vb:
            return True, True
        if not va and not vb:
            return False, False
        x = p["kp"][a if va else b][0]
        return x < cx, x >= cx

    def draw_guide(self, screen, raw):
        """Siluet transparan di tengah tiap bingkai: hijau = terlihat, merah = belum."""
        pulse = 0.5 + 0.5 * math.sin(time.time() * math.pi)
        alpha = 0.22 + 0.33 * pulse
        GREEN, RED = (70, 220, 70), (60, 60, 240)
        col = lambda ok: GREEN if ok else RED

        for i in range(2):
            p = raw[i]
            px, py = PANELS[i]
            cx = px + SW // 2
            P = lambda dx, dy: (int(cx + dx), int(py + dy))

            if p is not None:
                v = p["kp"][p["conf"] > 0.3][:, 0]
                mid = float(v.mean()) if len(v) else W / 2
                head = self._vis(p, 0)
                sh = self._pair(p, 5, 6, mid)
                el = self._pair(p, 7, 8, mid)
                ha = self._pair(p, 9, 10, mid)
                hp = self._pair(p, 11, 12, mid)
                kn = self._pair(p, 13, 14, mid)
                ft = self._pair(p, 15, 16, mid)
            else:
                head, sh, el, ha, hp, kn, ft = False, *([(False, False)] * 6)

            ov = screen.copy()
            cv2.circle(ov, P(0, 95), 40, col(head), -1, cv2.LINE_AA)
            cv2.fillPoly(ov, [np.array([P(-78, 175), P(78, 175), P(60, 255), P(-60, 255)])],
                         col(sh[0] and sh[1]))
            cv2.fillPoly(ov, [np.array([P(-60, 255), P(60, 255), P(50, 345), P(-50, 345)])],
                         col(hp[0] and hp[1]))

            def limb(a, b, ok):
                cv2.line(ov, a, b, col(ok), 28, cv2.LINE_AA)
                cv2.circle(ov, b, 14, col(ok), -1, cv2.LINE_AA)

            for s, k in ((0, -1), (1, 1)):
                limb(P(k * 78, 175), P(k * 112, 265), el[s])
                limb(P(k * 112, 265), P(k * 125, 355), ha[s])
                limb(P(k * 38, 340), P(k * 45, 460), kn[s])
                limb(P(k * 45, 460), P(k * 45, 575), ft[s])

            r = (slice(py, py + PAN_H), slice(px, px + SW))
            screen[r] = cv2.addWeighted(ov[r], alpha, screen[r], 1 - alpha, 0)

            ready = (head and sh[0] and sh[1] and el[0] and el[1] and hp[0] and hp[1] and kn[0] and kn[1] and ft[0] and ft[1])
            if ready:
                put(screen, "SIAP!", (cx, py + 25), 0.8, (0, 255, 0), 2, True)
            else:
                put(screen, "Wajib hijau semua: kepala sampai kaki", (cx, py + 25), 0.55, (255, 255, 255), 1, True)

    def paste_ref(self, screen, ref, scale=1.0):
        pw, ph = int(PW * scale), int(PH * scale)
        panel = ref["panel"] if scale == 1.0 else cv2.resize(ref["panel"], (pw, ph))
        x0, y0 = (W - pw) // 2, (H - ph) // 2 + 30
        screen[y0:y0 + ph, x0:x0 + pw] = panel
        cv2.rectangle(screen, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), 4)

    def missing_text(self, players):
        return [self.pname(i) for i in range(2) if players[i] is None]

    def warn(self, img, lines):
        if any(self.partial):
            names = ", ".join(self.pname(i) for i in range(2) if self.partial[i])
            lines = ["BADAN BELUM TERLIHAT!",
                     f"Mundur sampai seluruh badan (kepala sampai kaki) terlihat: {names}"]
        shade(img, 0, H - 175, W, H - 55, (0, 0, 160), 0.75)
        for k, ln in enumerate(lines):
            put(img, ln, (W // 2, H - 145 + k * 40), 1.0 if k == 0 else 0.7, (255, 255, 255), 2, True)

    # ---------- rekaman ----------
    def start_recording(self):
        self.session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(REC_DIR, exist_ok=True)
        self.snap_dir = os.path.join(SNAP_DIR, self.session)
        os.makedirs(self.snap_dir, exist_ok=True)
        self.rec_path = os.path.join(REC_DIR, f"game_{self.session}.mp4")
        self.writer = cv2.VideoWriter(self.rec_path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (W, H))
        self.rec_t0, self.rec_n = time.time(), 0

    def record(self, screen):
        if self.writer is None:
            return
        target = int((time.time() - self.rec_t0) * REC_FPS) + 1
        while self.rec_n < target:
            self.writer.write(screen)
            self.rec_n += 1

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def discard_recording(self):
        self.stop_recording()
        try:
            if self.rec_path and os.path.exists(self.rec_path):
                os.remove(self.rec_path)
            if self.snap_dir and os.path.isdir(self.snap_dir):
                os.rmdir(self.snap_dir)
        except OSError:
            pass

    # ---------- alur game ----------
    def reset_game(self):
        self.round = 0
        self.results = []
        self.ok_since = None

    def begin_play(self):
        if self.nrounds == 0:
            return
        self.reset_game()
        self.active = 0
        self.state = "NAMES"

    def start_game(self):
        random.shuffle(self.refs)                       # acak gerakan tiap permainan
        self.nrounds = min(ROUNDS, len(self.refs))
        self.reset_game()
        self.state = "WAIT"

    def begin_pose(self):
        """Mulai satu gerakan: suara 'mulai' + suara 5 detik pose."""
        self.state, self.t0 = "POSE", time.time()
        self.buf = [[], []]
        self.audio.play("mulai")
        self.audio.play("pose5")

    def next_round(self):
        self.round += 1
        if self.round >= self.nrounds:
            self.stop_recording()
            self.state = "FINAL"
        else:
            self.begin_pose()

    def totals(self):
        return [sum(r["scores"][i] for r in self.results) for i in range(2)]

    # ---------- layar ----------
    def s_menu(self, frame):
        screen = (frame * 0.35).astype(np.uint8)
        put(screen, "POSE BATTLE", (W // 2, 180), 3.0, GOLD, 5, True)
        put(screen, f"Adu gerakan 2 pemain! {POSE_TIME} detik per pose.",
            (W // 2, 275), 0.9, (255, 255, 255), 2, True)
        put(screen, "Pemain 1 berdiri di KIRI, Pemain 2 di KANAN. Mundur sampai seluruh badan (kepala sampai kaki) terlihat.",
            (W // 2, 320), 0.7, (220, 220, 220), 1, True)
        if self.nrounds == 0:
            put(screen, f"Belum ada gambar pose! Taruh gambar di folder '{POSE_DIR}' lalu jalankan ulang.",
                (W // 2, 365), 0.7, (80, 80, 255), 2, True)
        else:
            put(screen, f"Gambar pose terbaca: {len(self.refs)} (tiap game {self.nrounds} gerakan acak)",
                (W // 2, 365), 0.7, (0, 255, 0), 1, True)
        if not SENSOR_ON:
            put(screen, "MODE TEST: sensor mati, skor acak", (W // 2, 700), 0.7, (0, 165, 255), 2, True)
        self.btn_play.draw(screen, self.mouse)
        self.btn_quit.draw(screen, self.mouse)
        if self.click:
            if self.btn_play.hit(self.click):
                self.begin_play()
            elif self.btn_quit.hit(self.click):
                self.quit = True
        return screen

    def s_names(self, frame):
        screen = (frame * 0.35).astype(np.uint8)
        put(screen, "MASUKKAN NAMA PEMAIN", (W // 2, 110), 1.8, GOLD, 3, True)
        put(screen, "Klik kolom atau tekan TAB untuk pindah. ENTER = lanjut. Kosong = nama default.",
            (W // 2, 165), 0.65, (220, 220, 220), 1, True)

        blink = int(time.time() * 2) % 2 == 0
        for i, (x, y, w, h) in enumerate(self.field_rects):
            col = P_COLORS[i]
            act = self.active == i
            label = f"Pemain {i + 1} (berdiri di {'KIRI' if i == 0 else 'KANAN'})"
            put(screen, label, (x, y - 18), 0.8, col, 2, False)
            shade(screen, x, y, x + w, y + h, (255, 255, 255) if act else (0, 0, 0), 0.25 if act else 0.55)
            cv2.rectangle(screen, (x, y), (x + w, y + h), col, 4 if act else 2)
            txt = self.names[i]
            if txt:
                put(screen, txt, (x + 20, y + h // 2), 1.2, (255, 255, 255), 2, False, False)
            else:
                put(screen, f"Player {i + 1}", (x + 20, y + h // 2), 1.2, (140, 140, 140), 2, False, False)
            if act and blink:
                (tw, _), _ = cv2.getTextSize(txt, FONT, 1.2, 2)
                cx = x + 20 + tw + 6
                cv2.line(screen, (cx, y + 14), (cx, y + h - 14), (255, 255, 255), 3)
            put(screen, f"{len(txt)}/{NAME_MAX}", (x + w - 8, y + h + 22), 0.5, (200, 200, 200), 1, False, False)

        self.btn_menu.draw(screen, self.mouse)
        self.btn_start.draw(screen, self.mouse)
        if self.click:
            for i, (x, y, w, h) in enumerate(self.field_rects):
                if x <= self.click[0] <= x + w and y <= self.click[1] <= y + h:
                    self.active = i
            if self.btn_start.hit(self.click):
                self.start_game()
            elif self.btn_menu.hit(self.click):
                self.state = "MENU"
        return screen

    def names_key(self, key):
        if key == 255:
            return
        if key == 27:
            self.state = "MENU"
        elif key == 13:
            if self.active == 0:
                self.active = 1
            else:
                self.audio.play("klik")
                self.start_game()
        elif key == 9:
            self.active = 1 - self.active
        elif key in (8, 127):
            self.names[self.active] = self.names[self.active][:-1]
        elif 32 <= key <= 126 and len(self.names[self.active]) < NAME_MAX:
            self.names[self.active] += chr(key)

    def s_wait(self, frame):
        players = self.detect(frame)
        screen = self.compose(frame, players)
        self.draw_guide(screen, self.raw)            # siluet panduan hijau/merah
        shade(screen, 0, 0, W, 66, (0, 0, 0), 0.6)
        put(screen, f"{self.pname(0)} di KIRI, {self.pname(1)} di KANAN - berdiri di depan kamera",
            (W // 2, 22), 0.8, (255, 255, 255), 2, True)
        miss = self.missing_text(players)
        if not miss:
            if self.ok_since is None:
                self.ok_since = time.time()
            held = time.time() - self.ok_since
            put(screen, "Kedua pemain terdeteksi! Tahan posisi...", (W // 2, 50), 0.7, (0, 255, 0), 2, True)
            cv2.rectangle(screen, (0, 60), (int(W * min(held / 1.5, 1)), 66), (0, 255, 0), -1)
            if held >= 1.5:
                self.start_recording()
                self.last_cd = None
                self.state, self.t0 = "COUNTDOWN", time.time()
        else:
            self.ok_since = None
            self.warn(screen, ["PEMAIN TIDAK TERDETEKSI!",
                               f"Belum terlihat: {', '.join(miss)}  (terdeteksi {2 - len(miss)}/2). "
                               "Pastikan 2 orang berada di dalam kamera."])
        self.btn_cancel.draw(screen, self.mouse)
        if self.click and self.btn_cancel.hit(self.click):
            self.state = "MENU"
        return screen

    def s_countdown(self, frame):
        players = self.detect(frame)
        screen = self.compose(frame, players)
        self.paste_ref(screen, self.refs[0])
        t = time.time() - self.t0
        miss = self.missing_text(players)
        if miss:
            self.warn(screen, ["PEMAIN TIDAK TERDETEKSI!", f"Belum terlihat: {', '.join(miss)}"])
            self.audio.stop_sfx()
            self.discard_recording()
            self.ok_since = None
            self.state = "WAIT"
            return screen
        n = max(1, COUNTDOWN_TIME - int(t))
        if n != self.last_cd:                    # bunyi tiap ganti angka 3, 2, 1
            self.last_cd = n
            self.audio.play("hitung")
        shade(screen, 0, 0, W, H, (0, 0, 0), 0.35)
        put(screen, "BERSIAP!", (W // 2, 90), 1.6, GOLD, 4, True)
        put(screen, str(n), (W // 2, H // 2 + 30), 6.0, (255, 255, 255), 10, True)
        if t >= COUNTDOWN_TIME:
            self.begin_pose()
        return screen

    def s_pose(self, frame):
        players = self.detect(frame)
        clean = self.compose(frame, players, header=True)      # jepretan (tanpa gambar tengah)
        screen = clean.copy()
        ref = self.refs[self.round]
        remain = POSE_TIME - (time.time() - self.t0)

        self.paste_ref(screen, ref)

        frac = max(0, remain) / POSE_TIME
        cv2.rectangle(screen, (0, 0), (int(W * frac), 10), (0, int(255 * frac), int(255 * (1 - frac))), -1)
        put(screen, f"GERAKAN {self.round + 1}/{self.nrounds}", (W // 2, 40), 1.0, GOLD, 2, True)
        put(screen, str(max(1, math.ceil(remain))), (W // 2, 100), 2.2, (255, 255, 255), 5, True)

        tot = self.totals()
        for i, x in enumerate((PANELS[0][0] + SW // 2, PANELS[1][0] + SW // 2)):
            shade(screen, x - 130, H - 38, x + 130, H - 8, (0, 0, 0), 0.5)
            put(screen, f"{self.pname(i)}: {tot[i]:.1f}", (x, H - 23), 0.5, P_COLORS[i], 1, True)

        miss = self.missing_text(players)
        if miss:
            self.warn(screen, ["PEMAIN TIDAK TERDETEKSI!", f"Belum terlihat: {', '.join(miss)}"])

        if remain > SCORE_WINDOW:
            self.buf = [[], []]
        elif SENSOR_ON:
            for i in range(2):
                p = players[i]
                self.buf[i].append(0.0 if p is None else
                                   similarity(ref["kp"], ref["conf"], p["kp"], p["conf"]))

        if remain <= 0:                         # JEPRET + hitung skor
            self.audio.stop_sfx()               # hentikan suara 5 detik
            self.audio.play("jepret")
            if SENSOR_ON:
                scores = [round(float(np.median(b)), 1) if b else 0.0 for b in self.buf]
            else:
                scores = [round(random.uniform(40, 95), 1) for _ in range(2)]
            self.results.append({"name": ref["name"], "scores": scores,
                                 "detected": [p is not None for p in players],
                                 "snap": clean, "panel": ref["panel"],
                                 "thumb": cv2.resize(clean, (180, 101))})
            self.state, self.t0, self.saved = "RESULT", time.time(), False
        return screen

    def s_result(self, frame):
        r = self.results[-1]
        el = time.time() - self.t0
        screen = (r["snap"] * 0.7).astype(np.uint8)
        s1, s2 = r["scores"]
        win = 0 if s1 > s2 else 1 if s2 > s1 else -1

        put(screen, f"HASIL GERAKAN {len(self.results)}/{self.nrounds}: {r['name']}",
            (W // 2, 40), 1.0, (255, 255, 255), 2, True)
        self.paste_ref(screen, {"panel": r["panel"]})

        for i in range(2):
            cx = PANELS[i][0] + SW // 2
            col = GOLD if win == i else P_COLORS[i]
            shade(screen, cx - 135, H - 115, cx + 135, H - 45, (0, 0, 0), 0.75)
            cv2.rectangle(screen, (cx - 135, H - 115), (cx + 135, H - 45), col, 3 if win == i else 1)
            put(screen, self.pname(i), (cx, H - 96), 0.6, col, 1, True)
            if r["detected"][i]:
                put(screen, f"{r['scores'][i]:.1f}", (cx, H - 66), 0.8, col, 1, True)
            else:
                put(screen, "TIDAK TERDETEKSI", (cx, H - 66), 0.5, (80, 80, 255), 1, True)
            if win == i:
                put(screen, "MENANG!", (cx, H - 133), 0.8, GOLD, 2, True)

        label = "SERI!" if win == -1 else f"Menang: {self.pname(win)}"
        put(screen, label, (W // 2, 90), 0.9, GOLD, 2, True)
        left = max(0, math.ceil(RESULT_TIME - el))
        put(screen, f"Lanjut dalam {left} detik (SPASI = lanjut)",
            (W // 2, H - 25), 0.55, (255, 255, 255), 1, True)

        if not self.saved:
            cv2.imwrite(os.path.join(self.snap_dir, f"gerakan_{len(self.results)}.jpg"), screen)
            self.saved = True
        if el < 0.25:
            shade(screen, 0, 0, W, H, (255, 255, 255), 0.9 * (1 - el / 0.25))
        if el >= RESULT_TIME:
            self.next_round()
        return screen

    def s_final(self):
        screen = np.full((H, W, 3), (35, 28, 28), np.uint8)
        t = self.totals()
        put(screen, "GAME SELESAI", (W // 2, 50), 1.8, GOLD, 4, True)
        if t[0] == t[1]:
            put(screen, "HASIL SERI!", (W // 2, 110), 1.4, (255, 255, 255), 3, True)
        else:
            w = 0 if t[0] > t[1] else 1
            put(screen, f"PEMENANG: {self.pname(w).upper()}", (W // 2, 110), 1.5, P_COLORS[w], 3, True)
        put(screen, f"{self.pname(0)} {t[0]:.1f}   vs   {self.pname(1)} {t[1]:.1f}",
            (W // 2, 160), 1.0, (255, 255, 255), 2, True)

        cols = (430, 540)
        put(screen, "Gerakan", (60, 215), 0.8, (200, 200, 200), 1, False, False)
        for k, x in enumerate(cols):
            put(screen, self.pname(k)[:7], (x, 215), 0.6, P_COLORS[k], 2, False, False)
        for i, r in enumerate(self.results):
            y = 255 + i * 42
            put(screen, f"{i + 1}. {r['name'][:20]}", (60, y), 0.75, (255, 255, 255), 1, False, False)
            for k, x in enumerate(cols):
                best = r["scores"][k] == max(r["scores"]) and r["scores"][0] != r["scores"][1]
                put(screen, f"{r['scores'][k]:.1f}", (x, y), 0.75, GOLD if best else (255, 255, 255), 1, False, False)
        y = 255 + len(self.results) * 42 + 15
        cv2.line(screen, (50, y - 28), (640, y - 28), (150, 150, 150), 1)
        put(screen, "TOTAL", (60, y), 0.85, GOLD, 2, False, False)
        for k, x in enumerate(cols):
            put(screen, f"{t[k]:.1f}", (x, y), 0.85, GOLD, 2, False, False)

        for i, r in enumerate(self.results):
            x = 680 + (i % 3) * 192
            yy = 215 + (i // 3) * 150
            screen[yy:yy + 101, x:x + 180] = r["thumb"]
            put(screen, f"{i + 1}: {r['scores'][0]:.0f} vs {r['scores'][1]:.0f}",
                (x + 90, yy + 125), 0.6, (255, 255, 255), 1, True, False)

        put(screen, f"Rekaman: {self.rec_path}", (W // 2, 575), 0.6, (180, 180, 180), 1, True, False)
        for b in (self.btn_replay, self.btn_again, self.btn_tomenu, self.btn_exit):
            b.draw(screen, self.mouse)
        put(screen, "MAIN LAGI = pemain yang sama  |  MENU = ganti pemain",
            (W // 2, 705), 0.55, (180, 180, 180), 1, True, False)
        if self.click:
            if self.btn_replay.hit(self.click):
                self.pb = cv2.VideoCapture(self.rec_path)
                self.pb_frame, self.pb_last, self.state = None, 0, "PLAYBACK"
            elif self.btn_again.hit(self.click):
                self.start_game()                 # pemain & nama sama, langsung main lagi
            elif self.btn_tomenu.hit(self.click):
                self.names = ["", ""]             # kosongkan nama supaya bisa ganti pemain
                self.state = "MENU"
            elif self.btn_exit.hit(self.click):
                self.quit = True
        return screen

    def s_playback(self):
        now = time.time()
        if self.pb_frame is None or now - self.pb_last >= 1 / REC_FPS:
            ok, fr = self.pb.read()
            if not ok:
                self.pb.release()
                self.state = "FINAL"
                return self.s_final()
            self.pb_frame, self.pb_last = fr, now
        screen = self.pb_frame.copy()
        put(screen, "REKAMAN PERMAINAN", (200, 40), 0.9, (80, 80, 255), 2, True)
        self.btn_back.draw(screen, self.mouse)
        if self.click and self.btn_back.hit(self.click):
            self.pb.release()
            self.state = "FINAL"
        return screen

    # ---------- loop utama ----------
    def run(self):
        while not self.quit:
            ok, raw = self.cap.read()
            if not ok:
                break
            frame = fit_frame(cv2.flip(raw, 1))   # mirror seperti cermin

            if self.click and self.click_on_button():
                self.audio.play("klik")           # suara tekan tombol

            st = self.state
            if st == "MENU":
                screen = self.s_menu(frame)
            elif st == "NAMES":
                screen = self.s_names(frame)
            elif st == "WAIT":
                screen = self.s_wait(frame)
            elif st == "COUNTDOWN":
                screen = self.s_countdown(frame)
            elif st == "POSE":
                screen = self.s_pose(frame)
            elif st == "RESULT":
                screen = self.s_result(frame)
            elif st == "FINAL":
                screen = self.s_final()
            else:
                screen = self.s_playback()
            self.click = None

            self.audio.set_music(MUSIC_STATE.get(self.state))   # musik sesuai layar

            if st in ("COUNTDOWN", "POSE", "RESULT"):
                self.record(screen)

            cv2.imshow(WIN, screen)
            key = cv2.waitKey(1) & 0xFF

            if self.state == "NAMES":
                self.names_key(key)
            elif key == ord("q"):
                break
            elif key == 27:
                if self.state == "MENU":
                    break
                self.audio.stop_sfx()
                self.stop_recording()
                self.state = "MENU"
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            elif key in (13, ord("p")) and self.state == "MENU":
                self.audio.play("klik")
                self.begin_play()
            elif key == 32 and self.state == "RESULT":
                self.audio.play("klik")
                self.next_round()

        self.stop_recording()
        self.audio.close()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "cari":
        coord_helper(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "lihat":
        preview_poses()
    else:
        PoseBattle().run()