"""config.py - SEMUA PENGATURAN game ada di sini. Ubah angka di file ini saja."""
import os
import sys

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # jangan dibesarkan Windows (anti buram)
    except Exception:
        pass
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# ======================= UMUM =======================
W, H = 1280, 720
ROUNDS = 6                # jumlah gerakan tiap game
POSE_TIME = 5             # detik waktu berpose
COUNTDOWN_TIME = 3
RESULT_TIME = 5
REC_FPS = 20
MODEL_NAME = "yolov8n-pose.pt"
POSE_DIR, REC_DIR, SNAP_DIR = "poses", "recordings", "snapshots"
PW, PH = 320, 400         # ukuran gambar pose di tengah
NAME_MAX = 12
WIN = "Pose Battle"
SENSOR_ON = False         # True = sensor aktif | False = mode TEST (skor acak)

# ======================= SKOR =======================
SIGMA = 0.5               # makin kecil = penilaian makin ketat
ANGLE_TOL = 40.0
TILT_FORGIVE = 8.0        # selisih kemiringan bahu yang dimaafkan (derajat)
SCORE_WINDOW = 1.0        # skor diambil dari N detik terakhir (median)
ANGLE_FORGIVE = 12.0      # selisih arah anggota badan yang dianggap sempurna (derajat)
POS_FORGIVE = 0.15        # selisih posisi siku/tangan yang dianggap sempurna
BASE_CAP = 0.6            # batas pengurangan skor "diam"

# ======================= SENSOR =======================
CORE = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]   # bahu, siku, pinggul, lutut, kaki wajib terlihat
CONF_MIN = 0.5
MARGIN = 8
NAME2IDX = {
    "hidung": 0,
    "bahu_kiri": 6, "bahu_kanan": 5,
    "siku_kiri": 8, "siku_kanan": 7,
    "tangan_kiri": 10, "tangan_kanan": 9,
    "pinggul_kiri": 12, "pinggul_kanan": 11,
    "lutut_kiri": 14, "lutut_kanan": 13,
    "kaki_kiri": 16, "kaki_kanan": 15,
}

# ======================= SUARA =======================
SOUND_ON = True           # False = matikan semua suara
SOUND_DIR = "sounds"
SETTINGS_FILE = "settings.json"
MUSIC_VOL = 0.5
SFX_VOL = 0.9
SFX_FILES = {"klik": "klik", "hitung": "hitung", "mulai": "mulai",
             "pose5": "pose5detik", "jepret": "jepret"}
MUSIC_FILES = {"menu": "musik_menu", "main": "musik_main", "leaderboard": "musik_leaderboard"}
MUSIC_STATE = {   # musik yang diputar di tiap layar (None = tanpa musik)
    "SPLASH": "menu", "MENU": "menu", "NAMES": "menu", "WAIT": "menu", "SETTINGS": "menu",
    "COUNTDOWN": "main", "POSE": "main", "RESULT": "main",
    "FINAL": "leaderboard", "PLAYBACK": None,
}

# ======================= LAYOUT PANEL PEMAIN =======================
SW = 445
PT, PB = 70, 50
PAN_H = H - PT - PB
CROP_W = int(H * SW / PAN_H)
PANELS = [(5, PT), (W - 5 - SW, PT)]

# ======================= WARNA (urutan BGR!) =======================
P_COLORS = [(255, 200, 0), (180, 90, 255)]
GOLD = (0, 215, 255)
SKY_TOP, SKY_MID, SKY_BOT = (244, 155, 43), (255, 200, 124), (255, 235, 205)
NAVY, BLUE, GOLDC = (107, 60, 11), (192, 101, 21), (60, 201, 255)
GREEN_ON, GRAY_OFF = (50, 125, 46), (174, 164, 144)
CLOUDS = [(90, 70, 220, 70, 0.0), (1000, 160, 260, 80, 2.0), (150, 560, 180, 58, 4.0), (960, 600, 230, 70, 1.0)]