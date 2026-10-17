"""Utilitas bersama untuk semua skrip studi kasus."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend tanpa layar, aman dijalankan di server/CI

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_fig(fig, name: str) -> Path:
    """Simpan figure ke folder outputs/ dan kembalikan path-nya."""
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[gambar] tersimpan: {path.relative_to(ROOT)}")
    return path


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
