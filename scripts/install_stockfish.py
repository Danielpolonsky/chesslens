"""Download the official Stockfish binary for this platform into ./bin.

    python scripts/install_stockfish.py

ChessLens also finds Stockfish on PATH, at /usr/games/stockfish (Debian/Ubuntu
``apt install stockfish``, used on Streamlit Community Cloud via packages.txt) or at the
path in the STOCKFISH_PATH environment variable, so this script is optional.
Stockfish is GPL-3.0 licensed; the binary is downloaded, not redistributed with ChessLens.
"""

from __future__ import annotations

import io
import platform
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

RELEASE = "sf_19"
BASE = f"https://github.com/official-stockfish/Stockfish/releases/download/{RELEASE}"
ASSETS = {
    ("Windows", "AMD64"): "stockfish-windows-x86-64-universal.zip",
    ("Windows", "ARM64"): "stockfish-windows-arm64-universal.zip",
    ("Linux", "x86_64"): "stockfish-linux-x86-64-universal.tar.gz",
    ("Linux", "aarch64"): "stockfish-linux-arm64-universal.tar.gz",
    ("Darwin", "arm64"): "stockfish-macos-universal.tar.gz",
    ("Darwin", "x86_64"): "stockfish-macos-universal.tar.gz",
}
BIN = Path(__file__).resolve().parent.parent / "bin"


def main() -> int:
    key = (platform.system(), platform.machine())
    asset = ASSETS.get(key)
    if asset is None:
        print(f"No prebuilt binary configured for {key}; install Stockfish with your package manager.")
        return 1
    url = f"{BASE}/{asset}"
    print(f"Downloading {url}")
    data = requests.get(url, timeout=120).content
    BIN.mkdir(exist_ok=True)
    target = BIN / ("stockfish.exe" if key[0] == "Windows" else "stockfish")
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".exe"))
            target.write_bytes(zf.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            member = next(m for m in tf.getmembers()
                          if m.isfile() and Path(m.name).name.startswith("stockfish-"))
            target.write_bytes(tf.extractfile(member).read())
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
    print(f"Installed {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
