"""
Step 2 of ingestion: normalise every spec to .docx.

Older 3GPP specs ship as .doc, which is an OLE2 compound binary — painful to
parse from Python. .docx is just a ZIP of XML, which python-docx reads cleanly.
So we convert everything to .docx once, offline, with LibreOffice headless.

LibreOffice must be installed. Get it from https://www.libreoffice.org/download
On Windows it usually lands at:
    C:\\Program Files\\LibreOffice\\program\\soffice.exe
If it is not on PATH, set SOFFICE_PATH in your .env.

Usage
-----
    python -m src.ingest.convert
    python -m src.ingest.convert --force
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DOCX_DIR, secrets  # noqa: E402

WINDOWS_GUESSES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]
MAC_GUESSES = ["/Applications/LibreOffice.app/Contents/MacOS/soffice"]
LINUX_GUESSES = ["/usr/bin/soffice", "/usr/bin/libreoffice", "/snap/bin/libreoffice"]


def find_soffice() -> str | None:
    """Explicit config -> PATH -> platform guesses."""
    if secrets.SOFFICE_PATH and Path(secrets.SOFFICE_PATH).exists():
        return secrets.SOFFICE_PATH
    for name in ("soffice", "libreoffice", "soffice.exe"):
        found = shutil.which(name)
        if found:
            return found
    for guess in WINDOWS_GUESSES + MAC_GUESSES + LINUX_GUESSES:
        if Path(guess).exists():
            return guess
    return None


def convert_one(soffice: str, src: Path, out_dir: Path, timeout: int = 300) -> Path | None:
    """
    Convert a single .doc to .docx.

    Each call gets its own -env:UserInstallation profile directory. Without it,
    concurrent or repeated LibreOffice invocations fight over a shared profile
    lock and silently do nothing — a genuinely confusing failure mode.
    """
    with tempfile.TemporaryDirectory() as profile:
        profile_uri = Path(profile).absolute().as_uri()
        cmd = [
            soffice,
            f"-env:UserInstallation={profile_uri}",
            "--headless", "--norestore", "--invisible",
            "--convert-to", "docx",
            "--outdir", str(out_dir),
            str(src),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "HOME": profile},
            )
        except subprocess.TimeoutExpired:
            print(f"  ! timeout after {timeout}s: {src.name}")
            return None

    produced = out_dir / (src.stem + ".docx")
    if produced.exists() and produced.stat().st_size > 0:
        return produced

    print(f"  ! conversion produced nothing for {src.name}")
    if proc.stdout.strip():
        print(f"    stdout: {proc.stdout.strip()[:300]}")
    if proc.stderr.strip():
        print(f"    stderr: {proc.stderr.strip()[:300]}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert .doc specs to .docx")
    ap.add_argument("--force", action="store_true",
                    help="re-convert even if the .docx already exists")
    args = ap.parse_args()

    docs = sorted(DOCX_DIR.glob("*.doc"))
    if not docs:
        print(f"No .doc files in {DOCX_DIR} — nothing to convert.")
        existing = list(DOCX_DIR.glob("*.docx"))
        print(f"({len(existing)} .docx files already present.)")
        return 0

    soffice = find_soffice()
    if not soffice:
        print("ERROR: LibreOffice not found.")
        print("  Install it: https://www.libreoffice.org/download")
        print("  Or set SOFFICE_PATH in your .env, e.g.")
        print(r"  SOFFICE_PATH=C:\Program Files\LibreOffice\program\soffice.exe")
        return 1

    print(f"Using LibreOffice: {soffice}")
    print(f"Converting {len(docs)} .doc file(s)\n")

    ok, failed = 0, []
    for src in docs:
        target = DOCX_DIR / (src.stem + ".docx")
        if target.exists() and target.stat().st_size > 0 and not args.force:
            print(f"= {src.name} (already converted)")
            ok += 1
            continue

        print(f"> {src.name}")
        if convert_one(soffice, src, DOCX_DIR):
            print(f"  -> {target.name}")
            ok += 1
        else:
            failed.append(src.name)

    print(f"\nDone. {ok}/{len(docs)} converted.")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
