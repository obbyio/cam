# pose_points.py - titik sensor berproporsi MANUSIA (tangan + kaki), bahu ikut miring
# Nama kunci = nama file di folder poses/.  kiri / kanan = sisi pada GAMBAR.
#
#   buat(miring, (kiri_atas, kiri_bawah), (kanan_atas, kanan_bawah), kaki)
#   miring : kemiringan bahu (derajat). minus = kanan gambar lebih tinggi, plus = kanan lebih rendah
#   sudut  : 0 = lurus ke bawah, 90 = ke kanan gambar, 180 = ke atas, -90 = ke kiri gambar
#   kaki   : ((paha_kiri, betis_kiri), (paha_kanan, betis_kanan)), sudutnya sama seperti lengan
import math

CX, CY = 500, 400      # tengah bahu
SW = 200               # lebar bahu
UP = 0.85 * SW         # lengan atas
FORE = 0.70 * SW       # lengan bawah
TORSO = 1.2 * SW       # bahu ke pinggul
HIPW = 0.75 * SW       # lebar pinggul
THIGH = 1.6 * SW       # paha
SHIN = 1.6 * SW        # betis


def _seg(p, ang, ln):
    a = math.radians(ang)
    return (round(p[0] + ln * math.sin(a)), round(p[1] + ln * math.cos(a)))


def buat(miring, kiri, kanan, kaki=((-3, -3), (3, 3))):
    t = math.radians(miring)
    dx, dy = SW / 2 * math.cos(t), SW / 2 * math.sin(t)
    bk = (round(CX - dx), round(CY - dy))
    br = (round(CX + dx), round(CY + dy))
    sk = _seg(bk, kiri[0], UP)
    tk = _seg(sk, kiri[1], FORE)
    sr = _seg(br, kanan[0], UP)
    tr = _seg(sr, kanan[1], FORE)

    # pinggul: di bawah tengah bahu, tegak lurus garis bahu
    hx = CX + TORSO * -math.sin(t)
    hy = CY + TORSO * math.cos(t)
    hdx, hdy = HIPW / 2 * math.cos(t), HIPW / 2 * math.sin(t)
    pk = (round(hx - hdx), round(hy - hdy))
    pr = (round(hx + hdx), round(hy + hdy))
    lk = _seg(pk, kaki[0][0], THIGH)
    fk = _seg(lk, kaki[0][1], SHIN)
    lr = _seg(pr, kaki[1][0], THIGH)
    fr = _seg(lr, kaki[1][1], SHIN)
    return {
        "bahu_kiri": bk, "bahu_kanan": br,
        "siku_kiri": sk, "tangan_kiri": tk,
        "siku_kanan": sr, "tangan_kanan": tr,
        "pinggul_kiri": pk, "pinggul_kanan": pr,
        "lutut_kiri": lk, "kaki_kiri": fk,
        "lutut_kanan": lr, "kaki_kanan": fr,
    }


POSE_POINTS = {
    # Olaf: satu kaki berdiri, satu kaki ditendang ke kanan
    "Tangan V Ceria.jpg":         buat(-8,  (-137, -152), (114, 126),  ((5, 5), (70, 90))),
    # Masha: satu kaki berdiri, satu kaki diangkat ke belakang-kanan
    "Rentang Tangan Terbang.jpg": buat(-3,  (-114, -114), (105, 105),  ((-5, 0), (70, 115))),
    # Po: kuda-kuda lebar
    "Jurus Satu Tangan Naik.jpg": buat(-8,  (55, 70), (149, -133),     ((-55, -10), (40, 5))),
    # Daisy: kaki rapat
    "Sopan Tangan Bertaut.jpg":   buat(10,  (25, 50), (-8, 27),        ((5, 0), (-5, 0))),
    # Beruang: berdiri tegak
    "Pamer Otot.jpg":             buat(0,   (-70, 160), (73, -168),    ((-8, -3), (8, 3))),
    # Jerry: berdiri
    "Peluk Terbuka.jpg":          buat(0,   (-140, -125), (130, 120),  ((-3, 0), (3, 0))),
    # Naruto: berdiri sedikit terbuka
    "Segel Ninja.jpg":            buat(0,   (-25, 83), (25, -83),      ((-8, 0), (8, 0))),
    # Snoopy: satu kaki berdiri, satu kaki diangkat ke kanan
    "Balet Rentang Tangan.jpg":   buat(0,   (-88, -90), (85, 90),      ((5, 0), (70, 130))),
    # Kelinci: loncat
    "Teriak Tangan Naik.jpg":     buat(-12, (-112, -117), (150, 155),  ((5, 25), (25, 10))),
    # Monyet: satu kaki berdiri, satu kaki ditendang ke kanan
    "Hore Dua Tangan.jpg":        buat(0,   (-150, -170), (150, 170),  ((-8, 0), (80, 20))),
    # Po loncat: satu lutut terangkat
    "Angkat Tangan Tinggi.jpg":   buat(0,   (-155, -105), (155, 105),  ((-10, -5), (20, 0))),
    # Babi: satu kaki berdiri, satu kaki ditendang ke kiri
    "Satu Tangan Pinggang.jpg":   buat(-12, (-20, 45), (-172, -170),   ((-75, -110), (5, 0))),
    # Spider-Man: jongkok lompat
    "Lempar Jaring.jpg":          buat(25,  (-165, -155), (35, 65),    ((-110, 25), (85, 10))),
    # Dinosaurus: berdiri
    "Gaya T-Rex.jpg":             buat(0,   (-10, 160), (10, -160),    ((-10, 0), (10, 0))),
    # Balerina: satu kaki ditarik ke belakang-kiri, satu kaki berdiri
    "Balerina Tarik Kaki.jpg":    buat(-9,  (-71, -71), (120, 120),    ((-100, -150), (-10, 0))),
    # Ultraman: kuda-kuda lebar
    "Menunjuk Silang.jpg":        buat(-8,  (-20, 165), (-95, -90),    ((-25, 0), (30, 5))),
    # Peregangan samping: kaki membuka
    "Peregangan Samping.jpg":     buat(-77, (-13, -13), (176, 176),    ((-30, -25), (30, 40))),
    # Minion: tangan terbuka ke samping-bawah, satu kaki melangkah
    "Minion Jalan.jpg":       buat(0,  (-60, -60), (60, 60),    ((-35, 10), (5, 0))),
    # Ayam (tafsiran): dua siku ditekuk seperti sayap
    "Ayam Gemuk.jpg":         buat(0,  (-70, -150), (70, 150),  ((-5, 0), (5, 0))),
    # Yoga pohon: dua tangan menyatu di dada, satu lutut ditekuk ke samping
    "Yoga Pohon.jpg":         buat(0,  (-20, 155), (20, -155),  ((0, 0), (70, -50))),
    # Anak kepang: dua tangan terbuka ditekuk, satu kaki diangkat
    "Tari Kepang.jpg":        buat(-3, (-110, -145), (100, 125), ((-30, 25), (5, 0))),
    # Yoga penari: tangan kiri memegang kaki belakang, tangan kanan terentang
    "Yoga Penari.jpg":        buat(-3, (-100, -115), (110, 115), ((-105, -170), (3, 0))),
    # Pohon (tafsiran): dua tangan terangkat ke atas seperti dahan
    "Pohon Rindang.jpg":      buat(0,  (-150, -175), (150, 175), ((-3, 0), (3, 0))),
    # Kuda-kuda yoga: tangan lurus ke samping, kaki terbuka lebar
    "Yoga Kuda-Kuda.jpg":     buat(0,  (-92, -92), (92, 92),     ((-75, -5), (75, 5))),
}