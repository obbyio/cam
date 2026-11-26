"""audio.py - efek suara, musik latar, dan penyimpanan pengaturan suara."""
import json
import os

from config import *  # noqa: F401,F403  (config harus diimpor sebelum pygame)

try:
    import pygame
except ImportError:
    pygame = None
    print("[!] pygame belum terpasang (pip install pygame), game jalan tanpa suara.")


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        return bool(d.get("music", True)), bool(d.get("sfx", True))
    except Exception:
        return True, True


def save_settings(music, sfx):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump({"music": music, "sfx": sfx}, fh)
    except Exception:
        pass


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
        self.music_on, self.sfx_on = load_settings()
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
        if self.ok and self.sfx_on and key in self.sfx:
            self.sfx[key].play()

    def set_music(self, key):
        """Ganti musik latar (tidak mengulang dari awal kalau musiknya sama)."""
        if not self.ok or key == self.cur:
            return
        self.cur = key
        try:
            pygame.mixer.music.stop()
            if self.music_on and key and key in self.music:
                pygame.mixer.music.load(self.music[key])
                pygame.mixer.music.set_volume(MUSIC_VOL)
                pygame.mixer.music.play(-1)
        except Exception as e:
            print("[!] Musik gagal diputar:", e)

    def set_music_on(self, on):
        self.music_on = on
        self.cur = "?"                       # paksa musik layar ini dimuat ulang / dihentikan
        if self.ok and not on:
            pygame.mixer.music.stop()

    def set_sfx_on(self, on):
        self.sfx_on = on
        if self.ok and not on:
            pygame.mixer.stop()

    def stop_sfx(self):
        if self.ok:
            pygame.mixer.stop()

    def close(self):
        if self.ok:
            pygame.mixer.quit()