"""
POSE BATTLE - titik masuk game.  Jalankan:  python ver2.py

  python pose_battle.py                         -> main game
  python pose_battle.py cari poses/pose1.jpg    -> cari koordinat gambar
  python pose_battle.py lihat                   -> pratinjau skeleton pose

Struktur file:
  config.py      semua pengaturan (ubah angka di sini)
  audio.py       efek suara & musik
  scoring.py     rumus skor kemiripan
  ui.py          alat gambar tampilan (tombol, latar, logo, teks)
  references.py  memuat gambar pose + titik dari pose_points.py
  pose_points.py titik sensor tiap gambar (punyamu)
  game.py        isi game (layar, deteksi pemain, alur permainan)
  tools.py       alat bantu cari koordinat & pratinjau
Instal: pip install ultralytics opencv-python numpy pygame
"""
import sys

import config  # noqa: F401  (harus pertama: mengatur DPI Windows)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "cari":
        from tools import coord_helper
        coord_helper(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "lihat":
        from tools import preview_poses
        preview_poses()
    else:
        from game import PoseBattle
        PoseBattle().run()


if __name__ == "__main__":
    main()