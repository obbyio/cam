"""game.py - isi game: deteksi pemain, semua layar, dan alur permainan."""
import datetime
import math
import os
import random
import time

import cv2
import numpy as np
from ultralytics import YOLO

from config import *  # noqa: F401,F403
from audio import Audio, save_settings
from references import load_references
from scoring import body_visible, similarity
from ui import (Button, Sky, draw_logo, ease_back, fill_round, fit_frame, lerp_c, pill, put,
                shade, title_pose_battle, title_text, FONT)


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
        self.disp = (W, H)        # ukuran gambar yang benar-benar tampil di jendela
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.mouse, self.click = (0, 0), None
        self.fullscreen = False
        self.quit = False
        self.state = "SPLASH"
        self.sky = Sky()
        self.pending = None                 # aksi tombol yang menunggu animasi klik selesai
        self.last_state = None
        self.enter_t = time.time()
        self.sw_p = {}                      # posisi animasi saklar
        self.writer = None
        self.rec_path = None
        self.snap_dir = None
        self.pb = None
        self.pb_frame = None
        self.pb_last = 0
        self.partial = [False, False]
        self.raw = [None, None]
        self.buf = [[], []]
        self.last_cd = None
        self.saved = False
        self.names = ["", ""]
        self.active = 0
        self.cam_x = [W * 0.25, W * 0.75]

        self.btn_play = Button("PLAY", W // 2 - 220, 320, 440, 80, BLUE)
        self.btn_settings = Button("PENGATURAN", W // 2 - 220, 425, 440, 80, BLUE)
        self.btn_quit = Button("KELUAR", W // 2 - 220, 530, 440, 80, (40, 40, 198))
        self.btn_music = Button("", W // 2 + 138, 275 - 32, 132, 64, BLUE)    # area klik saklar
        self.btn_sfx = Button("", W // 2 + 138, 375 - 32, 132, 64, BLUE)
        self.btn_setback = Button("KEMBALI", W // 2 - 150, 500, 300, 76, BLUE)
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
        self.splash_t0 = time.time()

    # ---------- input ----------
    def on_mouse(self, event, x, y, flags, param):
        dw, dh = self.disp
        x, y = int(x * W / dw), int(y * H / dh)       # kembalikan ke koordinat 1280x720
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def click_on_button(self):
        """True jika klik terakhir mengenai tombol di layar yang sedang tampil."""
        bs = {
            "MENU": [self.btn_play, self.btn_settings, self.btn_quit],
            "SETTINGS": [self.btn_music, self.btn_sfx, self.btn_setback],
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

            ready = (head and sh[0] and sh[1] and el[0] and el[1] and hp[0] and hp[1]
                     and kn[0] and kn[1] and ft[0] and ft[1])
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

    def queue(self, fn, btn):
        """Jalankan aksi tombol setelah animasi klik (0,28 detik)."""
        btn.press_t = time.time()
        self.pending = (fn, time.time() + 0.28)

    # ---------- layar: splash, menu, pengaturan ----------
    def s_splash(self, frame):
        img = self.sky.draw()
        t = time.time() - self.splash_t0
        cx, cy = W // 2, 220
        p = ease_back((t - 0.1) / 0.9)                           # logo membesar sambil berputar
        rot = -math.radians(200) * (1 - min(1.0, t / 1.0))
        if t > 1.2:                                               # cincin berdenyut
            k = ((t - 1.2) % 2.0) / 2.0
            cv2.circle(img, (cx, cy), int(95 * (1 + 0.5 * k)), GOLDC, 4, cv2.LINE_AA)
        draw_logo(img, cx, cy, int(95 * p), rot)

        word, scale, th = "POSE BATTLE", 2.8, 6                  # huruf muncul satu per satu
        widths = [cv2.getTextSize(ch, FONT, scale, th)[0][0] for ch in word]
        x = W // 2 - sum(widths) // 2
        for i, ch in enumerate(word):
            k = ease_back((t - (0.5 + 0.08 * i)) / 0.5)
            if t >= 0.5 + 0.08 * i and ch != " ":
                title_text(img, ch, 0, 420 + (1 - k) * 60, scale, GOLDC if i >= 5 else (255, 255, 255),
                           th, False, x)
            x += widths[i]

        if t > 1.6:
            k = (t - 1.6) / 0.6
            cv2.putText(img, "Tiru gayanya, rebut skornya!", (W // 2 - 270, 505), FONT, 1.0,
                        lerp_c((255, 215, 150), NAVY, k), 2, cv2.LINE_AA)
        pill(img, W // 2 - 210, 560, 420, 20, (255, 255, 255))
        prog = min(1.0, max(0.0, (t - 0.5) / 2.8))
        if prog > 0.03:
            pill(img, W // 2 - 207, 563, int(414 * prog), 14, GOLDC)
        cv2.putText(img, "Klik untuk lewati", (W // 2 - 100, 615), FONT, 0.6, NAVY, 1, cv2.LINE_AA)

        if t > 3.8 or self.click:
            self.state = "MENU"
        return img

    def s_menu(self, frame):
        img = self.sky.draw()
        draw_logo(img, W // 2, 105, 58)
        title_pose_battle(img, 225, 2.6, 5)
        if self.nrounds == 0:
            cv2.putText(img, f"Belum ada gambar pose! Taruh gambar di folder '{POSE_DIR}'.",
                        (W // 2 - 400, 290), FONT, 0.8, (40, 40, 198), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, f"{len(self.refs)} gambar pose | tiap game {self.nrounds} gerakan acak",
                        (W // 2 - 280, 285), FONT, 0.7, NAVY, 1, cv2.LINE_AA)
        for b in (self.btn_play, self.btn_settings, self.btn_quit):
            b.draw(img, self.mouse)
        if not SENSOR_ON:
            cv2.putText(img, "MODE TEST: sensor mati, skor acak", (W // 2 - 210, 690),
                        FONT, 0.7, (0, 110, 230), 2, cv2.LINE_AA)
        if self.click:
            if self.btn_play.hit(self.click):
                self.queue(self.begin_play, self.btn_play)
            elif self.btn_settings.hit(self.click):
                self.queue(lambda: setattr(self, "state", "SETTINGS"), self.btn_settings)
            elif self.btn_quit.hit(self.click):
                self.queue(lambda: setattr(self, "quit", True), self.btn_quit)
        return img

    def draw_switch(self, img, key, on, x, y):
        """Saklar ON/OFF: bulatan meluncur dengan efek memantul, warna berganti halus."""
        tgt = 1.0 if on else 0.0
        pv = self.sw_p.setdefault(key, [tgt, 0.0])
        pv[1] = (pv[1] + (tgt - pv[0]) * 0.30) * 0.68            # pegas -> sedikit memantul
        pv[0] += pv[1]
        p = pv[0]
        pill(img, x, y, 132, 64, lerp_c(GRAY_OFF, GREEN_ON, p), NAVY, 4)
        kx = int(x + 34 + 64 * p)
        cv2.circle(img, (kx, y + 35), 24, (120, 90, 60), -1, cv2.LINE_AA)
        cv2.circle(img, (kx, y + 32), 24, (255, 255, 255), -1, cv2.LINE_AA)
        if on:
            cv2.putText(img, "ON", (x + 20, y + 42), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "OFF", (x + 66, y + 42), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def s_settings(self, frame):
        img = self.sky.draw()
        title_text(img, "PENGATURAN", W // 2, 120, 2.2, (255, 255, 255), 5)
        a = self.audio
        cx0, cy0, cw, ch = W // 2 - 310, 200, 620, 250
        fill_round(img, cx0, cy0 + 10, cw, ch, 46, NAVY)
        fill_round(img, cx0, cy0, cw, ch, 46, (255, 255, 255))
        fill_round(img, cx0 + 5, cy0 + 5, cw - 10, ch - 10, 41, (255, 248, 240))
        for x in range(cx0 + 40, cx0 + cw - 40, 24):
            cv2.line(img, (x, cy0 + 125), (x + 12, cy0 + 125), (242, 203, 156), 3)
        for i, (lab, on, btn, key) in enumerate((("Musik", a.music_on, self.btn_music, "music"),
                                                  ("Efek Suara", a.sfx_on, self.btn_sfx, "sfx"))):
            cv2.putText(img, lab, (cx0 + 50, cy0 + 87 + i * 100), FONT, 1.3, NAVY, 2, cv2.LINE_AA)
            self.draw_switch(img, key, on, btn.x, btn.y)
        if not a.ok:
            cv2.putText(img, "Suara tidak aktif (pygame belum terpasang atau SOUND_ON = False)",
                        (W // 2 - 400, 660), FONT, 0.6, (40, 40, 198), 1, cv2.LINE_AA)
        self.btn_setback.draw(img, self.mouse)
        if self.click:
            if self.btn_music.hit(self.click):
                a.set_music_on(not a.music_on)
                save_settings(a.music_on, a.sfx_on)
            elif self.btn_sfx.hit(self.click):
                a.set_sfx_on(not a.sfx_on)
                save_settings(a.music_on, a.sfx_on)
            elif self.btn_setback.hit(self.click):
                self.queue(lambda: setattr(self, "state", "MENU"), self.btn_setback)
        return img

    # ---------- layar: input nama ----------
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

    # ---------- layar: cek sensor, hitung mundur, bermain ----------
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
                                   similarity(ref["kp"], ref["conf"], p["kp"], p["conf"], ref["base"]))

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

    # ---------- layar: hasil, leaderboard, rekaman ----------
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

        if not self.saved:                      # simpan dulu, baru kilat putih (supaya file tidak putih)
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

            if self.pending:                       # tunggu animasi klik selesai, abaikan klik lain
                self.click = None
                if time.time() >= self.pending[1]:
                    fn = self.pending[0]
                    self.pending = None
                    fn()
            if self.state != self.last_state:
                self.last_state = self.state
                self.enter_t = time.time()

            st = self.state
            if st == "SPLASH":
                screen = self.s_splash(frame)
            elif st == "MENU":
                screen = self.s_menu(frame)
            elif st == "NAMES":
                screen = self.s_names(frame)
            elif st == "SETTINGS":
                screen = self.s_settings(frame)
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

            el = time.time() - self.enter_t
            if st in ("SPLASH", "MENU", "SETTINGS") and el < 0.35:
                shade(screen, 0, 0, W, H, (255, 255, 255), 0.9 * (1 - el / 0.35))

            try:                                    # perbesar jendela dengan tajam (anti buram)
                _, _, ww, wh = cv2.getWindowImageRect(WIN)
                if ww > 0 and wh > 0:
                    self.disp = (ww, wh)
                    if (ww, wh) != (W, H):
                        screen = cv2.resize(screen, (ww, wh), interpolation=(
                            cv2.INTER_LINEAR if ww * wh > W * H else cv2.INTER_AREA))
            except Exception:
                pass
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