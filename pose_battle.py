"""
POSE BATTLE - Game adu gerakan 2 pemain dengan webcam
Deteksi pose : YOLOv8-Pose (ultralytics)
Kemiripan    : dihitung dengan NumPy (posisi sendi + sudut anggota badan)

Alur: MENU -> isi NAMA -> FOTO karakter -> cek pemain -> 6 gerakan -> hasil + rekaman
Kontrol: klik tombol / ENTER, SPASI = ambil foto / lanjut ronde, TAB = pindah kolom nama,
         F = fullscreen, ESC = kembali ke menu, Q = keluar
Gambar contoh sendiri: taruh foto orang berpose di folder "poses/" (jpg/png)
"""
import datetime
import glob
import math
import os
import time
import random

import cv2
import numpy as np
from ultralytics import YOLO

# ======================= PENGATURAN =======================
W, H = 1280, 720
ROUNDS = 6
POSE_TIME = 5
COUNTDOWN_TIME = 3
PHOTO_COUNTDOWN = 3       # hitung mundur foto karakter
RESULT_TIME = 5
REC_FPS = 20
MODEL_NAME = "yolov8n-pose.pt"
POSE_DIR, REC_DIR, SNAP_DIR = "poses", "recordings", "snapshots"
SIGMA = 0.7
PW, PH = 320, 400
MAX_NAME = 12             # maksimal huruf nama
WIN = "Pose Battle"
FONT = cv2.FONT_HERSHEY_DUPLEX
P_COLORS = [(255, 200, 0), (180, 90, 255)]
GOLD = (0, 215, 255)

BODY = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14),
         (14, 16), (5, 11), (6, 12), (5, 6), (11, 12)]
EDGES = LIMBS + [(0, 5), (0, 6)]


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


def similarity(ref_kp, ref_conf, kp, conf):
    """Skor 0-100 = 60% kemiripan posisi sendi + 40% kemiripan arah anggota badan."""
    r, rv = normalize(ref_kp, ref_conf)
    p, pv = normalize(kp, conf)
    if r is None or p is None:
        return 0.0

    idx = np.array([i for i in BODY if rv[i]])
    dist = np.linalg.norm(r[idx] - p[idx], axis=1)
    pos_score = (np.exp(-(dist / SIGMA) ** 2) * pv[idx]).mean()

    ang = []
    for a, b in LIMBS:
        if not (rv[a] and rv[b]):
            continue
        if not (pv[a] and pv[b]):
            ang.append(0.0)
            continue
        vr, vp = r[b] - r[a], p[b] - p[a]
        nr, npp = np.linalg.norm(vr), np.linalg.norm(vp)
        if nr < 1e-6:
            continue
        if npp < 1e-6:
            ang.append(0.0)
            continue
        cos = float(np.dot(vr, vp) / (nr * npp))
        ang.append(max(cos, 0.0) ** 3)
    ang_score = float(np.mean(ang)) if ang else 0.0

    return float(100 * (0.6 * pos_score + 0.4 * ang_score))


# ======================= GAMBAR CONTOH =======================
def make_pose(l_arm, r_arm, l_leg, r_leg):
    kp = np.zeros((17, 2))
    kp[0] = (0, -0.8)
    kp[1], kp[2] = (0.15, -0.9), (-0.15, -0.9)
    kp[3], kp[4] = (0.35, -0.85), (-0.35, -0.85)
    kp[5], kp[6] = (0.5, 0), (-0.5, 0)
    kp[11], kp[12] = (0.35, 1.9), (-0.35, 1.9)

    def chain(start, angs, lens):
        pts, cur = [], np.array(start, float)
        for a, l in zip(angs, lens):
            cur = cur + np.array([math.sin(math.radians(a)), math.cos(math.radians(a))]) * l
            pts.append(cur.copy())
        return pts

    kp[7], kp[9] = chain(kp[5], l_arm, (0.9, 0.8))
    kp[8], kp[10] = chain(kp[6], r_arm, (0.9, 0.8))
    kp[13], kp[15] = chain(kp[11], l_leg, (1.0, 1.0))
    kp[14], kp[16] = chain(kp[12], r_leg, (1.0, 1.0))
    return kp


SYNTH_POSES = [
    ("Huruf Y", make_pose((150, 150), (-150, -150), (10, 10), (-10, -10))),
    ("T-Pose", make_pose((90, 90), (-90, -90), (0, 0), (0, 0))),
    ("Satu Tangan Naik", make_pose((170, 170), (-10, -10), (8, 8), (-8, -8))),
    ("Kuda-kuda", make_pose((70, 170), (-70, -170), (35, 5), (-35, -5))),
    ("Satu Kaki Naik", make_pose((90, 90), (-90, -90), (75, 5), (-5, -5))),
    ("Tangan di Kepala", make_pose((100, -125), (-100, 125), (5, 5), (-5, -5))),
]


def make_synth_panel(name, kp):
    card = np.full((PH, PW, 3), 245, np.uint8)
    s, cx, sy = 52, PW // 2, 125
    px = lambda p: (int(cx + p[0] * s), int(sy + p[1] * s))
    col = (60, 60, 60)
    for a, b in [(5, 6), (6, 12), (12, 11), (11, 5)]:
        cv2.line(card, px(kp[a]), px(kp[b]), col, 8, cv2.LINE_AA)
    cv2.line(card, px((0, 0)), px((0, -0.5)), col, 8, cv2.LINE_AA)
    cv2.circle(card, px((0, -0.95)), int(0.42 * s), col, 6, cv2.LINE_AA)
    for a, b in [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]:
        cv2.line(card, px(kp[a]), px(kp[b]), (200, 120, 20), 9, cv2.LINE_AA)
    for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        cv2.circle(card, px(kp[i]), 7, (30, 30, 200), -1, cv2.LINE_AA)
    put(card, name, (PW // 2, PH - 28), 0.9, (40, 40, 40), 2, True, False)
    return card


def make_image_panel(img, name):
    card = np.full((PH, PW, 3), 245, np.uint8)
    h, w = img.shape[:2]
    f = min((PW - 20) / w, (PH - 70) / h)
    im = cv2.resize(img, (int(w * f), int(h * f)))
    x, y = (PW - im.shape[1]) // 2, 10
    card[y:y + im.shape[0], x:x + im.shape[1]] = im
    put(card, name[:22], (PW // 2, PH - 28), 0.9, (40, 40, 40), 2, True, False)
    return card


def load_references(model):
    refs = []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files += glob.glob(os.path.join(POSE_DIR, ext))
    for f in sorted(files):
        img = cv2.imread(f)
        if img is None:
            continue
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
        if best is None:
            print(f"[!] Orang tidak terdeteksi di {f}, dilewati.")
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        refs.append({"name": name, "kp": best[0], "conf": best[1],
                     "panel": make_image_panel(img, name)})
    for name, kp in SYNTH_POSES:
        refs.append({"name": name, "kp": kp, "conf": np.ones(17),
                     "panel": make_synth_panel(name, kp)})
    return refs[:ROUNDS]


# ======================= UTILITAS TAMPILAN =======================
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


# ---------- avatar (foto karakter) ----------
def make_avatar(crop, color, D=200):
    if crop is None or crop.size == 0:
        crop = np.full((D, D, 3), 90, np.uint8)
    face = cv2.resize(crop, (D, D))
    mask = np.zeros((D, D), np.uint8)
    cv2.circle(mask, (D // 2, D // 2), D // 2 - 3, 255, -1)
    return {"img": face, "mask": mask, "color": color}


def draw_avatar(screen, av, cx, cy, d, color=None, thick=4):
    x1, y1 = cx - d // 2, cy - d // 2
    if x1 < 0 or y1 < 0 or x1 + d > screen.shape[1] or y1 + d > screen.shape[0]:
        return
    if av is None:
        cv2.circle(screen, (cx, cy), d // 2 - 3, (90, 90, 90), -1, cv2.LINE_AA)
        cv2.circle(screen, (cx, cy), d // 2 - 3, color or (255, 255, 255), thick, cv2.LINE_AA)
        return
    face = cv2.resize(av["img"], (d, d))
    mask = cv2.resize(av["mask"], (d, d), interpolation=cv2.INTER_NEAREST) > 0
    roi = screen[y1:y1 + d, x1:x1 + d]
    roi[mask] = face[mask]
    cv2.circle(screen, (cx, cy), d // 2 - 3, color or av["color"], thick, cv2.LINE_AA)


class Button:
    def __init__(self, label, x, y, w, h, color):
        self.label, self.x, self.y, self.w, self.h, self.color = label, x, y, w, h, color

    def hit(self, p):
        return p is not None and self.x <= p[0] <= self.x + self.w and self.y <= p[1] <= self.y + self.h

    def draw(self, img, mouse):
        c = tuple(min(255, int(v * 1.25)) for v in self.color) if self.hit(mouse) else self.color
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), c, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 3)
        put(img, self.label, (self.x + self.w // 2, self.y + self.h // 2), 1.1, (255, 255, 255), 2, True)


# ======================= GAME =======================
class PoseBattle:
    def __init__(self):
        print("Memuat model...")
        self.model = YOLO(MODEL_NAME)
        self.model(np.zeros((640, 640, 3), np.uint8), verbose=False)
        self.refs = load_references(self.model)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise SystemExit("Webcam tidak ditemukan.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.mouse, self.click = (0, 0), None
        self.key = 255
        self.fullscreen = False
        self.quit = False
        self.state = "MENU"
        self.writer = None
        self.rec_path = None
        self.pb = None
        self.pb_frame = None
        self.pb_last = 0

        # data pemain
        self.names = ["Player 1", "Player 2"]
        self.inputs = ["", ""]
        self.field = 0
        self.avatars = [None, None]
        self.photo_t0 = None

        self.btn_play = Button("PLAY", W // 2 - 160, 400, 320, 80, (60, 170, 60))
        self.btn_quit = Button("KELUAR", W // 2 - 160, 500, 320, 70, (60, 60, 170))
        self.btn_cancel = Button("BATAL", 40, 630, 200, 60, (60, 60, 170))
        self.btn_replay = Button("PUTAR REKAMAN", 150, 620, 330, 70, (170, 110, 40))
        self.btn_again = Button("MAIN LAGI", 510, 620, 280, 70, (60, 170, 60))
        self.btn_exit = Button("KELUAR", 820, 620, 280, 70, (60, 60, 170))
        self.btn_back = Button("KEMBALI", W - 240, H - 90, 200, 60, (60, 60, 170))
        self.btn_next = Button("LANJUT", W // 2 - 160, 530, 320, 80, (60, 170, 60))
        self.btn_snap = Button("AMBIL FOTO", W // 2 - 170, 130, 340, 70, (60, 170, 60))
        self.btn_retake = Button("ULANG FOTO", W // 2 - 340, 600, 320, 80, (170, 110, 40))
        self.btn_start = Button("MULAI GAME", W // 2 + 20, 600, 320, 80, (60, 170, 60))
        self.reset_game()

    # ---------- input ----------
    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def name_box(self, i):
        return (W // 2 - 250, 240 + i * 130, 500, 70)   # x, y, w, h

    def handle_typing(self, key):
        if key == 255:
            return
        if key == 27:
            self.state = "MENU"
        elif key == 9:
            self.field = 1 - self.field
        elif key in (8, 127):
            self.inputs[self.field] = self.inputs[self.field][:-1]
        elif key == 13:
            if self.field == 0:
                self.field = 1
            else:
                self.confirm_names()
        elif 32 <= key < 127 and len(self.inputs[self.field]) < MAX_NAME:
            self.inputs[self.field] += chr(key)

    def confirm_names(self):
        self.names = [(self.inputs[i].strip() or f"Player {i + 1}") for i in range(2)]
        self.photo_t0 = None
        self.state = "PHOTO"

    # ---------- deteksi ----------
    def detect(self, frame):
        """Kembalikan [P1, P2]; P1 = orang di kiri layar, P2 = orang di kanan."""
        slots = [None, None]
        res = self.model(frame, imgsz=640, conf=0.4, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return slots
        xy = res.keypoints.xy.cpu().numpy()
        cf = res.keypoints.conf
        cf = cf.cpu().numpy() if cf is not None else np.ones(xy.shape[:2])
        people = []
        for kp, c in zip(xy, cf):
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
        return slots

    def draw_players(self, img, players):
        for i, p in enumerate(players):
            if p is None:
                continue
            kp, c, col = p["kp"], p["conf"], P_COLORS[i]
            for a, b in EDGES:
                if c[a] > 0.3 and c[b] > 0.3:
                    cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), col, 4, cv2.LINE_AA)
            for j in range(17):
                if c[j] > 0.3:
                    cv2.circle(img, tuple(kp[j].astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)
            v = kp[c > 0.3]
            put(img, self.names[i], (int(v[:, 0].mean()), int(v[:, 1].min()) - 25), 1.0, col, 2, True)

    def warn(self, img, lines):
        shade(img, 0, H - 130, W, H - 20, (0, 0, 160), 0.75)
        for k, ln in enumerate(lines):
            put(img, ln, (W // 2, H - 100 + k * 40), 1.0 if k == 0 else 0.7, (255, 255, 255), 2, True)

    def missing_text(self, players):
        return [self.names[i] for i in range(2) if players[i] is None]

    def capture_avatars(self, frame, players):
        avs = []
        for i, p in enumerate(players):
            kp = p["kp"]
            sw = float(np.linalg.norm(kp[5] - kp[6]))
            side = int(min(max(80, 0.8 * sw), H, W))
            cx, cy = kp[0][0], kp[0][1] - 0.08 * sw
            x1 = int(np.clip(cx - side / 2, 0, W - side))
            y1 = int(np.clip(cy - side / 2, 0, H - side))
            avs.append(make_avatar(frame[y1:y1 + side, x1:x1 + side].copy(), P_COLORS[i]))
        return avs

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

    # ---------- alur game ----------
    def reset_game(self):
        self.round = 0
        self.results = []
        self.ok_since = None

    def begin_play(self):
        """Dari menu: isi nama dulu."""
        self.reset_game()
        self.inputs = [n if n not in ("Player 1", "Player 2") else "" for n in self.names]
        self.field = 0
        self.state = "NAMES"

    def restart(self):
        """Main lagi dengan nama & foto yang sama."""
        self.reset_game()
        self.state = "WAIT"

    def next_round(self):
        self.round += 1
        if self.round >= ROUNDS:
            self.stop_recording()
            self.state = "FINAL"
        else:
            self.state = "POSE"
            self.t0 = time.time()

    def totals(self):
        return [sum(r["scores"][i] for r in self.results) for i in range(2)]

    # ---------- layar ----------
    def s_menu(self, frame):
        screen = (frame * 0.35).astype(np.uint8)
        put(screen, "POSE BATTLE", (W // 2, 190), 3.0, GOLD, 5, True)
        put(screen, f"Adu gerakan 2 pemain! {ROUNDS} gerakan, {POSE_TIME} detik per pose.",
            (W // 2, 285), 0.9, (255, 255, 255), 2, True)
        put(screen, "Isi nama, foto karakter, lalu tirukan gerakan di layar. Mundur sampai badan terlihat.",
            (W // 2, 330), 0.7, (220, 220, 220), 1, True)
        self.btn_play.draw(screen, self.mouse)
        self.btn_quit.draw(screen, self.mouse)
        if self.click:
            if self.btn_play.hit(self.click):
                self.begin_play()
            elif self.btn_quit.hit(self.click):
                self.quit = True
        return screen

    def s_names(self, frame):
        screen = (frame * 0.3).astype(np.uint8)
        put(screen, "MASUKKAN NAMA PEMAIN", (W // 2, 110), 1.8, GOLD, 4, True)
        put(screen, "Ketik nama lalu tekan ENTER. TAB / klik kotak untuk pindah kolom.",
            (W // 2, 170), 0.7, (220, 220, 220), 1, True)
        labels = ["Nama Player 1 (berdiri di KIRI)", "Nama Player 2 (berdiri di KANAN)"]
        for i in range(2):
            x, y, w, h = self.name_box(i)
            if self.click and x <= self.click[0] <= x + w and y <= self.click[1] <= y + h:
                self.field = i
            active = self.field == i
            put(screen, labels[i], (x, y - 15), 0.8, P_COLORS[i], 2, False)
            cv2.rectangle(screen, (x, y), (x + w, y + h), (45, 45, 45), -1)
            cv2.rectangle(screen, (x, y), (x + w, y + h), P_COLORS[i], 5 if active else 2)
            txt = self.inputs[i]
            if txt == "" and not active:
                put(screen, f"Player {i + 1}", (x + 20, y + 48), 1.0, (120, 120, 120), 1, False, False)
            else:
                cur = "|" if active and int(time.time() * 2) % 2 == 0 else ""
                put(screen, txt + cur, (x + 20, y + 48), 1.1, (255, 255, 255), 2, False, False)
        self.btn_next.draw(screen, self.mouse)
        put(screen, "ESC = kembali ke menu", (W // 2, 650), 0.6, (180, 180, 180), 1, True, False)
        if self.click and self.btn_next.hit(self.click):
            self.confirm_names()
        return screen

    def s_photo(self, frame):
        players = self.detect(frame)
        screen = frame.copy()
        self.draw_players(screen, players)
        ready = all(p is not None and p["conf"][0] > 0.3 for p in players)

        shade(screen, 0, 0, W, 110, (0, 0, 0), 0.55)
        put(screen, "FOTO KARAKTER", (W // 2, 35), 1.2, GOLD, 2, True)
        put(screen, f"{self.names[0]} di KIRI, {self.names[1]} di KANAN. Hadap kamera, wajah terlihat jelas.",
            (W // 2, 80), 0.8, (255, 255, 255), 2, True)

        if self.photo_t0 is None:
            if ready:
                self.btn_snap.draw(screen, self.mouse)
                put(screen, "Siap! Klik AMBIL FOTO atau tekan SPASI", (W // 2, 230), 0.9, (0, 255, 0), 2, True)
                if (self.click and self.btn_snap.hit(self.click)) or self.key == 32:
                    self.photo_t0 = time.time()
            else:
                miss = self.missing_text(players)
                self.warn(screen, ["PEMAIN TIDAK TERDETEKSI!",
                                   f"Belum terlihat: {', '.join(miss) if miss else 'wajah belum jelas'}. "
                                   "Pastikan 2 orang berada di dalam kamera."])
        else:
            remain = PHOTO_COUNTDOWN - (time.time() - self.photo_t0)
            put(screen, str(max(1, math.ceil(remain))), (W // 2, H // 2), 6.0, (255, 255, 255), 10, True)
            put(screen, "Senyum!", (W // 2, H // 2 - 130), 1.8, GOLD, 4, True)
            if remain <= 0:
                if ready:
                    self.avatars = self.capture_avatars(frame, players)
                    self.state = "PHOTO_OK"
                    self.flash_t0 = time.time()
                self.photo_t0 = None

        self.btn_cancel.draw(screen, self.mouse)
        if self.click and self.btn_cancel.hit(self.click):
            self.state = "NAMES"
        return screen

    def s_photo_ok(self, frame):
        screen = (frame * 0.3).astype(np.uint8)
        put(screen, "KARAKTER SIAP!", (W // 2, 90), 1.8, GOLD, 4, True)
        for i, cx in enumerate((W // 4, 3 * W // 4)):
            draw_avatar(screen, self.avatars[i], cx, 330, 280, thick=8)
            put(screen, self.names[i], (cx, 510), 1.4, P_COLORS[i], 3, True)
            put(screen, f"Player {i + 1}", (cx, 555), 0.7, (200, 200, 200), 1, True, False)
        el = time.time() - getattr(self, "flash_t0", 0)
        if el < 0.25:
            shade(screen, 0, 0, W, H, (255, 255, 255), 0.9 * (1 - el / 0.25))
        self.btn_retake.draw(screen, self.mouse)
        self.btn_start.draw(screen, self.mouse)
        if self.click:
            if self.btn_retake.hit(self.click):
                self.photo_t0 = None
                self.state = "PHOTO"
            elif self.btn_start.hit(self.click):
                self.reset_game()
                self.state = "WAIT"
        return screen

    def s_wait(self, frame):
        players = self.detect(frame)
        screen = frame.copy()
        self.draw_players(screen, players)
        shade(screen, 0, 0, W, 110, (0, 0, 0), 0.55)
        put(screen, f"Berdiri di depan kamera: {self.names[0]} di KIRI, {self.names[1]} di KANAN",
            (W // 2, 45), 1.0, (255, 255, 255), 2, True)
        miss = self.missing_text(players)
        if not miss:
            if self.ok_since is None:
                self.ok_since = time.time()
            held = time.time() - self.ok_since
            put(screen, "Kedua pemain terdeteksi! Tahan posisi...", (W // 2, 85), 0.9, (0, 255, 0), 2, True)
            cv2.rectangle(screen, (0, 105), (int(W * min(held / 1.5, 1)), 115), (0, 255, 0), -1)
            if held >= 1.5:
                self.start_recording()
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
        screen = frame.copy()
        self.draw_players(screen, players)
        t = time.time() - self.t0
        shade(screen, 0, 0, W, H, (0, 0, 0), 0.35)
        put(screen, "BERSIAP!", (W // 2, H // 2 - 120), 2.0, GOLD, 4, True)
        put(screen, str(max(1, COUNTDOWN_TIME - int(t))), (W // 2, H // 2 + 20), 6.0, (255, 255, 255), 10, True)
        miss = self.missing_text(players)
        if miss:
            self.warn(screen, ["PEMAIN TIDAK TERDETEKSI!", f"Belum terlihat: {', '.join(miss)}"])
        if t >= COUNTDOWN_TIME:
            self.state, self.t0 = "POSE", time.time()
        return screen

    def s_pose(self, frame):
        players = self.detect(frame)
        clean = frame.copy()
        self.draw_players(clean, players)
        screen = clean.copy()
        ref = self.refs[self.round]
        remain = POSE_TIME - (time.time() - self.t0)

        x0, y0 = (W - PW) // 2, (H - PH) // 2 + 30
        roi = screen[y0:y0 + PH, x0:x0 + PW]
        screen[y0:y0 + PH, x0:x0 + PW] = cv2.addWeighted(roi, 0.25, ref["panel"], 0.75, 0)
        cv2.rectangle(screen, (x0, y0), (x0 + PW, y0 + PH), (255, 255, 255), 3)

        frac = max(0, remain) / POSE_TIME
        cv2.rectangle(screen, (0, 0), (int(W * frac), 12), (0, int(255 * frac), int(255 * (1 - frac))), -1)
        put(screen, f"GERAKAN {self.round + 1}/{ROUNDS}", (W // 2, 45), 1.2, GOLD, 2, True)
        put(screen, str(max(1, math.ceil(remain))), (W // 2, 120), 3.0, (255, 255, 255), 6, True)

        tot = self.totals()
        for i, (ax, tx) in enumerate(((55, 215), (W - 55, W - 215))):
            draw_avatar(screen, self.avatars[i], ax, 62, 76)
            put(screen, self.names[i], (tx, 45), 0.8, P_COLORS[i], 2, True)
            put(screen, f"Skor {tot[i]:.1f}", (tx, 85), 0.8, (255, 255, 255), 2, True)

        miss = self.missing_text(players)
        if miss:
            self.warn(screen, ["PEMAIN TIDAK TERDETEKSI!", f"Belum terlihat: {', '.join(miss)}"])

        if remain <= 0:
            scores = []
            for i in range(2):
                p = players[i]
                scores.append(0.0 if p is None else
                              round(similarity(ref["kp"], ref["conf"], p["kp"], p["conf"]), 1))
            self.results.append({"name": ref["name"], "scores": scores,
                                 "detected": [p is not None for p in players],
                                 "snap": clean, "panel": ref["panel"],
                                 "thumb": cv2.resize(clean, (180, 101))})
            self.state, self.t0, self.saved = "RESULT", time.time(), False
        return screen

    def s_result(self, frame):
        r = self.results[-1]
        el = time.time() - self.t0
        screen = (r["snap"] * 0.55).astype(np.uint8)
        s1, s2 = r["scores"]
        win = 0 if s1 > s2 else 1 if s2 > s1 else -1

        put(screen, f"HASIL GERAKAN {len(self.results)}/{ROUNDS}: {r['name']}",
            (W // 2, 40), 1.1, (255, 255, 255), 2, True)
        for i, cx in enumerate((W * 0.2, W * 0.8)):
            cx = int(cx)
            col = GOLD if win == i else P_COLORS[i]
            shade(screen, cx - 190, 75, cx + 190, 285, (0, 0, 0), 0.65)
            cv2.rectangle(screen, (cx - 190, 75), (cx + 190, 285), col, 4 if win == i else 2)
            draw_avatar(screen, self.avatars[i], cx - 130, 112, 64, col, 3)
            put(screen, self.names[i], (cx + 35, 112), 0.9, P_COLORS[i], 2, True)
            if r["detected"][i]:
                put(screen, f"{r['scores'][i]:.1f}", (cx, 182), 2.6, col, 5, True)
                cv2.rectangle(screen, (cx - 150, 225), (cx + 150, 245), (80, 80, 80), -1)
                cv2.rectangle(screen, (cx - 150, 225), (cx - 150 + int(3 * r["scores"][i]), 245), col, -1)
                put(screen, "kemiripan (0-100)", (cx, 268), 0.6, (220, 220, 220), 1, True, False)
            else:
                put(screen, "TIDAK TERDETEKSI", (cx, 180), 1.0, (80, 80, 255), 2, True)
            if win == i:
                put(screen, "MENANG!", (cx, 310), 1.1, GOLD, 3, True)

        th = cv2.resize(r["panel"], (int(PW * 0.65), int(PH * 0.65)))
        x0, y0 = (W - th.shape[1]) // 2, (H - th.shape[0]) // 2 + 40
        screen[y0:y0 + th.shape[0], x0:x0 + th.shape[1]] = th
        cv2.rectangle(screen, (x0, y0), (x0 + th.shape[1], y0 + th.shape[0]), (255, 255, 255), 3)

        label = "SERI!" if win == -1 else f"Pemenang gerakan ini: {self.names[win]}"
        put(screen, label, (W // 2, H - 75), 1.0, GOLD, 2, True)
        left = max(0, math.ceil(RESULT_TIME - el))
        put(screen, f"Lanjut dalam {left} detik (SPASI = langsung lanjut)",
            (W // 2, H - 35), 0.7, (255, 255, 255), 1, True)

        if el < 0.25:
            shade(screen, 0, 0, W, H, (255, 255, 255), 0.9 * (1 - el / 0.25))
        if not self.saved:
            cv2.imwrite(os.path.join(self.snap_dir, f"gerakan_{len(self.results)}.jpg"), screen)
            self.saved = True
        if el >= RESULT_TIME:
            self.next_round()
        return screen

    def s_final(self):
        screen = np.full((H, W, 3), (35, 28, 28), np.uint8)
        t = self.totals()
        w = -1 if t[0] == t[1] else (0 if t[0] > t[1] else 1)

        put(screen, "GAME SELESAI", (W // 2, 50), 1.8, GOLD, 4, True)
        if w == -1:
            put(screen, "HASIL SERI!", (W // 2, 110), 1.4, (255, 255, 255), 3, True)
        else:
            put(screen, f"PEMENANG: {self.names[w]}", (W // 2, 110), 1.3, P_COLORS[w], 3, True)
        put(screen, f"{self.names[0]} {t[0]:.1f}   vs   {self.names[1]} {t[1]:.1f}",
            (W // 2, 160), 0.9, (255, 255, 255), 2, True)
        for i, ax in enumerate((130, W - 130)):
            draw_avatar(screen, self.avatars[i], ax, 95, 110, GOLD if w == i else None, 6 if w == i else 4)
            put(screen, self.names[i], (ax, 175), 0.8, P_COLORS[i], 2, True)

        put(screen, "Gerakan", (60, 225), 0.8, (200, 200, 200), 1, False, False)
        put(screen, self.names[0][:7], (440, 225), 0.7, P_COLORS[0], 2, False, False)
        put(screen, self.names[1][:7], (550, 225), 0.7, P_COLORS[1], 2, False, False)
        for i, r in enumerate(self.results):
            y = 265 + i * 40
            put(screen, f"{i + 1}. {r['name'][:20]}", (60, y), 0.75, (255, 255, 255), 1, False, False)
            for k, x in enumerate((440, 550)):
                best = r["scores"][k] == max(r["scores"]) and r["scores"][0] != r["scores"][1]
                put(screen, f"{r['scores'][k]:.1f}", (x, y), 0.75, GOLD if best else (255, 255, 255), 1, False, False)
        y = 265 + len(self.results) * 40 + 15
        cv2.line(screen, (50, y - 28), (640, y - 28), (150, 150, 150), 1)
        put(screen, "TOTAL", (60, y), 0.85, GOLD, 2, False, False)
        put(screen, f"{t[0]:.1f}", (440, y), 0.85, GOLD, 2, False, False)
        put(screen, f"{t[1]:.1f}", (550, y), 0.85, GOLD, 2, False, False)

        for i, r in enumerate(self.results):
            x = 680 + (i % 3) * 192
            yy = 235 + (i // 3) * 150
            screen[yy:yy + 101, x:x + 180] = r["thumb"]
            put(screen, f"{i + 1}: {r['scores'][0]:.0f} vs {r['scores'][1]:.0f}",
                (x + 90, yy + 125), 0.6, (255, 255, 255), 1, True, False)

        put(screen, f"Rekaman: {self.rec_path}", (W // 2, 580), 0.6, (180, 180, 180), 1, True, False)
        for b in (self.btn_replay, self.btn_again, self.btn_exit):
            b.draw(screen, self.mouse)
        if self.click:
            if self.btn_replay.hit(self.click):
                self.pb = cv2.VideoCapture(self.rec_path)
                self.pb_frame, self.pb_last, self.state = None, 0, "PLAYBACK"
            elif self.btn_again.hit(self.click):
                self.restart()
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
            frame = fit_frame(cv2.flip(raw, 1))
            st = self.state
            if st == "MENU":
                screen = self.s_menu(frame)
            elif st == "NAMES":
                screen = self.s_names(frame)
            elif st == "PHOTO":
                screen = self.s_photo(frame)
            elif st == "PHOTO_OK":
                screen = self.s_photo_ok(frame)
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

            if st in ("COUNTDOWN", "POSE", "RESULT"):
                self.record(screen)

            cv2.imshow(WIN, screen)
            key = cv2.waitKey(1) & 0xFF
            self.key = key

            if self.state == "NAMES":              # sedang mengetik nama
                self.handle_typing(key)
                continue
            if key == ord("q"):
                break
            elif key == 27:
                if self.state == "MENU":
                    break
                self.stop_recording()
                self.state = "MENU"
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            elif key in (13, ord("p")) and self.state == "MENU":
                self.begin_play()
            elif key == 32 and self.state == "RESULT":
                self.next_round()

        self.stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    PoseBattle().run()