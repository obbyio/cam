
"""ui.py - alat gambar tampilan: teks, tombol pil, latar langit, logo, judul."""
import math
import time
 
import cv2
import numpy as np
 
from config import *  # noqa: F401,F403
 
FONT = cv2.FONT_HERSHEY_DUPLEX
 
 
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
 
 
def lerp_c(a, b, t):
    t = min(1.0, max(0.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
 
 
def ease_back(t):
    """Membesar sedikit melewati target lalu kembali (efek memantul)."""
    t = min(1.0, max(0.0, t))
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
 
 
def fill_pill(img, x, y, w, h, col):
    r = h // 2
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), col, -1)
    cv2.circle(img, (x + r, y + r), r, col, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - r, y + r), r, col, -1, cv2.LINE_AA)
 
 
def pill(img, x, y, w, h, col, border=None, bw=0):
    x, y, w, h = int(x), int(y), int(w), int(h)
    if border is not None:
        fill_pill(img, x, y, w, h, border)
        fill_pill(img, x + bw, y + bw, w - 2 * bw, h - 2 * bw, col)
    else:
        fill_pill(img, x, y, w, h, col)
 
 
def fill_round(img, x, y, w, h, r, col):
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), col, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), col, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
 
 
def title_text(img, s, cx, cy, scale=2.0, fill=(255, 255, 255), th=5, center=True, x=None):
    """Teks judul: isi putih/emas, garis tepi navy, bayangan di bawah."""
    (tw, tt), _ = cv2.getTextSize(s, FONT, scale, th)
    x0 = int(cx - tw / 2) if center else int(x)
    y0 = int(cy + tt / 2)
    cv2.putText(img, s, (x0, y0 + 8), FONT, scale, NAVY, th + 9, cv2.LINE_AA)
    cv2.putText(img, s, (x0, y0), FONT, scale, NAVY, th + 9, cv2.LINE_AA)
    cv2.putText(img, s, (x0, y0), FONT, scale, fill, th, cv2.LINE_AA)
 
 
def title_pose_battle(img, cy, scale, th):
    (w1, _), _ = cv2.getTextSize("POSE ", FONT, scale, th)
    (w2, _), _ = cv2.getTextSize("BATTLE", FONT, scale, th)
    x = W // 2 - (w1 + w2) // 2
    title_text(img, "POSE ", 0, cy, scale, (255, 255, 255), th, False, x)
    title_text(img, "BATTLE", 0, cy, scale, GOLDC, th, False, x + w1)
 
 
def draw_logo(img, cx, cy, r, rot=0.0):
    """Logo: lingkaran biru bergambar orang berpose."""
    if r < 4:
        return
    cv2.circle(img, (cx, cy + int(r * 0.07)), r + 2, NAVY, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r * 0.9), BLUE, -1, cv2.LINE_AA)
    k = r / 95.0
    c, s = math.cos(rot), math.sin(rot)
    P = lambda px, py: (int(cx + (px * c - py * s) * k), int(cy + (px * s + py * c) * k))
    th = max(2, int(9 * k))
    for a, b in (((0, -15), (0, 17)), ((0, -7), (-27, -27)), ((0, -7), (27, -27)),
                 ((0, 17), (-17, 45)), ((0, 17), (17, 45))):
        cv2.line(img, P(*a), P(*b), (255, 255, 255), th, cv2.LINE_AA)
    cv2.circle(img, P(0, -31), max(2, int(13 * k)), GOLDC, -1, cv2.LINE_AA)
 
 
class Sky:
    """Latar langit biru polos dengan garis diagonal yang bergerak dan awan melayang.
    Pola digambar SEKALI (lebih lebar 120 px); tiap frame cukup menggeser potongannya (ringan)."""
 
    def __init__(self):
        t = np.linspace(0, 1, H)
        col = np.zeros((H, 3))
        for c in range(3):
            col[:, c] = np.interp(t, [0, 0.55, 1], [SKY_TOP[c], SKY_MID[c], SKY_BOT[c]])
        ww = W + 120
        base = np.repeat(col[:, None, :], ww, axis=1).astype(np.uint8)
        light = (base.astype(np.float32) * 0.86 + 255 * 0.14).astype(np.uint8)
        yy, xx = np.mgrid[0:H, 0:ww]
        mask = (((xx + yy) // 60) & 1) == 0
        self.wide = np.where(mask[..., None], light, base).astype(np.uint8)
 
    def draw(self):
        t = time.time()
        off = int((t * 40) % 120)
        img = self.wide[:, off:off + W].copy()
        for bx, by, w, h, ph in CLOUDS:
            m = math.sin(t * 0.9 + ph)
            fill_pill(img, int(bx + 24 * m), int(by - 14 * m), w, h, (255, 250, 245))
        return img
 
 
class Button:
    """Tombol pil: berubah emas + naik saat kursor di atasnya, memantul saat diklik."""
 
    def __init__(self, label, x, y, w, h, color):
        self.label, self.x, self.y, self.w, self.h, self.color = label, x, y, w, h, color
        self.hov = 0.0
        self.press_t = -1.0
 
    def hit(self, p):
        return p is not None and self.x <= p[0] <= self.x + self.w and self.y <= p[1] <= self.y + self.h
 
    def draw(self, img, mouse):
        over = self.hit(mouse)
        self.hov += ((1.0 if over else 0.0) - self.hov) * 0.25
        hv = self.hov
        sc, dy = 1 + 0.04 * hv, -6 * hv
        pt = time.time() - self.press_t if self.press_t > 0 else 9
        press = 0.0
        if pt < 0.3:
            press = math.sin(pt / 0.3 * math.pi)
            sc *= 1 - 0.10 * press
            dy += 8 * press
        w2, h2 = int(self.w * sc), int(self.h * sc)
        x2 = self.x + (self.w - w2) // 2
        y2 = self.y + (self.h - h2) // 2 + int(dy)
        fill = lerp_c(self.color, GOLDC, hv)
        shadow = tuple(int(v * 0.5) for v in self.color)
        pill(img, x2, y2 + int(9 - 7 * press + 5 * hv), w2, h2, shadow)
        pill(img, x2, y2, w2, h2, fill, (255, 255, 255), 5)
        tcol = lerp_c((255, 255, 255), NAVY, hv)
        (tw, th), _ = cv2.getTextSize(self.label, FONT, 1.0, 2)
        cv2.putText(img, self.label, (x2 + (w2 - tw) // 2, y2 + (h2 + th) // 2),
                    FONT, 1.0, tcol, 2, cv2.LINE_AA)