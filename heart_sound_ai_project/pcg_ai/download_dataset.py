"""Download and unpack the public PhysioNet/CinC Challenge 2016 training set.

The archive is downloaded from the official PhysioNet host. This script does not
redistribute the full dataset inside the project ZIP.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = [
    "https://physionet.org/files/challenge-2016/1.0.0/training.zip?download",
    "https://physionet.org/files/challenge-2016/1.0.0/training.zip",
]
DEFAULT_DEST = Path("data/physionet2016")
ARCHIVE_NAME = "training.zip"


def download_with_progress(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HeartSoundScienceFairPrototype/1.0 (educational project)"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        received = 0
        with temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if total:
                    pct = received * 100 / total
                    print(f"\rDownloading: {received / 1024**2:.1f} MB / {total / 1024**2:.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\rDownloaded {received / 1024**2:.1f} MB", end="", flush=True)
        print()
        if received < 1024:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Download was unexpectedly small; the server may have returned an error page.")
        temporary.replace(target)


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP integrity check failed at {bad_member}")
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != root and root not in member_path.parents:
                raise RuntimeError(f"Unsafe path inside ZIP: {member.filename}")
        archive.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official PhysioNet 2016 heart-sound training data.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination folder (default: data/physionet2016)")
    parser.add_argument("--keep-zip", action="store_true", help="Keep training.zip after extraction")
    args = parser.parse_args()

    dest = args.dest.expanduser().resolve()
    existing_refs = list(dest.rglob("REFERENCE.csv")) if dest.exists() else []
    if existing_refs:
        print(f"Dataset already appears to be extracted: {dest}")
        print(f"Found {len(existing_refs)} REFERENCE.csv file(s). No download needed.")
        return 0

    archive = dest.parent / ARCHIVE_NAME
    errors = []
    if not archive.exists():
        for url in URLS:
            try:
                print(f"Source: {url}")
                download_with_progress(url, archive)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                archive.with_suffix(archive.suffix + ".part").unlink(missing_ok=True)
        else:
            print("Could not download the dataset from PhysioNet.", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            print("You can download training.zip manually from https://physionet.org/content/challenge-2016/1.0.0/ and rerun this script.", file=sys.stderr)
            return 2

    try:
        print(f"Checking and extracting {archive} ...")
        safe_extract(archive, dest)
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 3

    refs = list(dest.rglob("REFERENCE.csv"))
    wavs = list(dest.rglob("*.wav"))
    if not refs or not wavs:
        print("Extraction finished, but labeled WAV files were not found. Check the archive layout.", file=sys.stderr)
        return 4

    note = dest / "DATASET_ATTRIBUTION.txt"
    note.write_text(
        "Dataset: PhysioNet/Computing in Cardiology Challenge 2016, version 1.0.0\n"
        "Source: https://physionet.org/content/challenge-2016/1.0.0/\n"
        "License: Open Data Commons Attribution License v1.0 (ODC-BY 1.0).\n"
        "Please retain this attribution and consult the dataset page/license before sharing data or derivatives.\n\n"
        "Citation: Liu C, Springer D, Li Q, Moody B, Juan RA, Chorro FJ, Castells F, Roig JM,\n"
        "Silva I, Johnson AE, Syed Z, Schmidt SE, Papadaniil CD, Hadjileontiadis L, Naseri H,\n"
        "Moukadem A, Dieterlen A, Brandt C, Tang H, Samieinasab M, Sameni R, Sameni MR, Mark RG,\n"
        "Clifford GD. An open access database for the evaluation of heart sound algorithms.\n"
        "Physiological Measurement. 2016;37(12):2181-2213.\n"
        "https://doi.org/10.1088/0967-3334/37/12/2181\n",
        encoding="utf-8",
    )
    if not args.keep_zip:
        archive.unlink(missing_ok=True)
    print(f"Success: found {len(wavs)} WAV recordings and {len(refs)} REFERENCE.csv files.")
    print(f"Dataset folder: {dest}")
    print("The dataset is for research/education; it is not a clinical diagnosis dataset for this device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
