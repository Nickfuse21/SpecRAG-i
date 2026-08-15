"""
Step 1 of ingestion: fetch 3GPP specs from the open FTP archive.

3GPP publishes every spec at a predictable URL:

    https://www.3gpp.org/ftp/Specs/archive/<series>_series/<spec>/<code>.zip

where <code> looks like  38331-i80  and the three characters after the dash
encode the version x.y.z as base-36-ish digits:

    0-9  ->  0..9
    a-z  -> 10..35

So  38331-i80  =>  i=18, 8=8, 0=0  =>  TS 38.331 V18.8.0

The FIRST digit tracks the Release, which is why we can pin a release just by
filtering on it:   f=15  g=16  h=17  i=18  j=19

Usage
-----
    python -m src.ingest.download                 # full corpus, Rel-18
    python -m src.ingest.download --minimal       # the 6-spec fallback set
    python -m src.ingest.download --release 17
    python -m src.ingest.download --spec 38.331   # just one, for testing
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DOCX_DIR, FTP_BASE, RAW_DIR, SPECS, SPECS_MINIMAL, TARGET_RELEASE  # noqa: E402

# 3gpp.org rejects the default python-requests user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
TIMEOUT = 60


# --------------------------------------------------------------------------
# version code helpers
# --------------------------------------------------------------------------
def decode_digit(ch: str) -> int:
    """0-9 -> 0..9,  a-z -> 10..35."""
    ch = ch.lower()
    if ch.isdigit():
        return int(ch)
    if "a" <= ch <= "z":
        return ord(ch) - ord("a") + 10
    raise ValueError(f"bad version digit: {ch!r}")


def decode_version(code: str) -> tuple[int, int, int]:
    """'i80' -> (18, 8, 0)"""
    if len(code) != 3:
        raise ValueError(f"version code must be 3 chars, got {code!r}")
    return tuple(decode_digit(c) for c in code)  # type: ignore[return-value]


def version_string(code: str) -> str:
    x, y, z = decode_version(code)
    return f"{x}.{y}.{z}"


@dataclass
class SpecFile:
    spec: str          # "38.331"
    code: str          # "i80"
    filename: str      # "38331-i80.zip"
    url: str

    @property
    def version(self) -> str:
        return version_string(self.code)

    @property
    def release(self) -> int:
        return decode_version(self.code)[0]

    @property
    def stem(self) -> str:
        return self.filename.removesuffix(".zip")   # "38331-i80"


# --------------------------------------------------------------------------
# listing + selection
# --------------------------------------------------------------------------
def spec_dir_url(spec: str) -> str:
    series = spec.split(".")[0]
    return f"{FTP_BASE}/{series}_series/{spec}/"


def list_versions(spec: str, session: requests.Session) -> list[SpecFile]:
    """Scrape the FTP directory listing for every published version of a spec."""
    url = spec_dir_url(spec)
    r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()

    flat = spec.replace(".", "")               # 38.331 -> 38331
    pattern = re.compile(rf"({re.escape(flat)}-([0-9a-zA-Z]{{3}}))\.zip", re.I)

    out: dict[str, SpecFile] = {}
    for stem, code in pattern.findall(r.text):
        code = code.lower()
        try:
            decode_version(code)
        except ValueError:
            continue
        fn = f"{flat}-{code}.zip"
        out[fn] = SpecFile(spec=spec, code=code, filename=fn, url=url + fn)
    return list(out.values())


def pick_version(candidates: list[SpecFile], release: int) -> SpecFile | None:
    """Highest y.z within the target release. Falls back to the newest overall."""
    if not candidates:
        return None
    in_release = [c for c in candidates if c.release == release]
    if in_release:
        return max(in_release, key=lambda c: decode_version(c.code))
    newest = max(candidates, key=lambda c: decode_version(c.code))
    print(f"  ! no Rel-{release} version for {newest.spec}; "
          f"newest available is V{newest.version} (Rel-{newest.release})")
    return newest


# --------------------------------------------------------------------------
# download + extract
# --------------------------------------------------------------------------
def download(sf: SpecFile, session: requests.Session) -> Path:
    dest = RAW_DIR / sf.filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = already have {sf.filename}")
        return dest

    with session.get(sf.url, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  {sf.filename}", leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)
    return dest


def extract(zip_path: Path, sf: SpecFile) -> Path | None:
    """Pull the largest .doc/.docx out of the zip into data/docx/."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.infolist()
            if not m.is_dir() and m.filename.lower().endswith((".doc", ".docx"))
        ]
        if not members:
            print(f"  ! no .doc/.docx inside {zip_path.name} "
                  f"(contains: {[m.filename for m in zf.namelist()[:5]]})")
            return None
        member = max(members, key=lambda m: m.file_size)
        ext = Path(member.filename).suffix.lower()
        dest = DOCX_DIR / f"{sf.stem}{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        with zf.open(member) as src, open(dest, "wb") as out:
            out.write(src.read())
        return dest


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Download 3GPP specs from the FTP archive")
    ap.add_argument("--release", type=int, default=TARGET_RELEASE)
    ap.add_argument("--minimal", action="store_true",
                    help="only the 6-spec fallback set (use if Day 1 runs long)")
    ap.add_argument("--spec", action="append", default=None,
                    help="download only this spec (repeatable), e.g. --spec 38.331")
    args = ap.parse_args()

    if args.spec:
        wanted = args.spec
    elif args.minimal:
        wanted = SPECS_MINIMAL
    else:
        wanted = [s for s, _ in SPECS]

    titles = dict(SPECS)
    session = requests.Session()
    manifest: list[str] = []
    failures: list[str] = []

    print(f"Downloading {len(wanted)} specs, pinned to Rel-{args.release}\n")

    for spec in wanted:
        print(f"{spec}  {titles.get(spec, '')}")
        try:
            candidates = list_versions(spec, session)
        except Exception as e:                                   # noqa: BLE001
            print(f"  ! listing failed: {e}")
            failures.append(spec)
            continue

        if not candidates:
            print("  ! no versions found — check the spec number")
            failures.append(spec)
            continue

        sf = pick_version(candidates, args.release)
        assert sf is not None
        print(f"  -> V{sf.version}  ({sf.filename})")

        try:
            zp = download(sf, session)
            doc = extract(zp, sf)
        except Exception as e:                                   # noqa: BLE001
            print(f"  ! download/extract failed: {e}")
            failures.append(spec)
            continue

        if doc:
            print(f"  -> {doc.name}")
            manifest.append(f"{spec}\t{sf.version}\tRel-{sf.release}\t{doc.name}")

    (RAW_DIR / "manifest.tsv").write_text(
        "spec\tversion\trelease\tfile\n" + "\n".join(manifest) + "\n", encoding="utf-8"
    )

    print(f"\nDone. {len(manifest)}/{len(wanted)} specs fetched.")
    print(f"Manifest: {RAW_DIR / 'manifest.tsv'}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        print("Re-run to retry — completed downloads are skipped.")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
