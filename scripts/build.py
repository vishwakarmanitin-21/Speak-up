"""Build FlowAI into a standalone .exe using PyInstaller.

Usage:
    python scripts/build.py

Output:
    dist/FlowAI.exe
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "FlowAI.spec"
DIST = ROOT / "dist"


def main() -> None:
    # Ensure pyinstaller is installed
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[Build] Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Clean previous build artifacts
    for d in (ROOT / "build", DIST):
        if d.exists():
            print(f"[Build] Cleaning {d}...")
            shutil.rmtree(d)

    # Run PyInstaller
    print("[Build] Building FlowAI.exe...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"],
        cwd=str(ROOT),
    )

    if result.returncode != 0:
        print("[Build] FAILED — see errors above.")
        sys.exit(1)

    exe_path = DIST / "FlowAI.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n[Build] SUCCESS: {exe_path}  ({size_mb:.1f} MB)")
        print("\nTo distribute:")
        print(f"  1. Copy {exe_path} to the target machine")
        print("  2. On first run, the user will be prompted for their OpenAI API key")
        print("  3. Config files (config.json, .env) are created next to the .exe")
    else:
        print("[Build] FAILED — exe not found in dist/")
        sys.exit(1)


if __name__ == "__main__":
    main()
