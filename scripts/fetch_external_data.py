"""Fetch third-party power system datasets into data/external/.

These are *not* vendored into the repository: they are large (RTS-GMLC is
~290 MB) and their licenses differ from gridlock's, so each stays in its
own upstream clone under ``data/external/`` (gitignored) with provenance
recorded here.

    python scripts/fetch_external_data.py            # everything
    python scripts/fetch_external_data.py rts-gmlc   # one dataset
    python scripts/fetch_external_data.py --list

Licensing, as verified 2026-08-04 — read this before redistributing
anything derived from these:

- **RTS-GMLC** carries no SPDX license file, but its README contains an
  explicit NREL "Data Use Disclaimer Agreement" granting "the right,
  without any fee or cost, to use, copy, and distribute these Data for any
  purpose whatsoever, provided that this entire notice appears in all
  copies", with attribution and an indemnity clause. Redistributable with
  the notice attached; automated license scanners will still flag it.
- **PowerSystemsTestData** (the NREL-118 mirror) has **no license file at
  all**, so it is all-rights-reserved by default. Fine to compute with
  locally; do not redistribute it or data derived from it.

Note that NREL was renamed the National Laboratory of the Rockies and
``nrel.gov`` no longer resolves; the original NREL-118 download URLs are
dead, which is why the Sienna mirror is used here.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "data" / "external"


@dataclass
class Dataset:
    """One upstream repository and how much of it we need."""

    key: str
    directory: str
    url: str
    summary: str
    license_note: str
    # Restrict the working tree to these paths (git sparse-checkout). Empty
    # means take everything.
    sparse_paths: list[str] = field(default_factory=list)


DATASETS = [
    Dataset(
        key="rts-gmlc",
        directory="RTS-GMLC",
        url="https://github.com/GridMod/RTS-GMLC.git",
        summary=(
            "73 buses, 158 generators (73 thermal), 120 AC + 1 DC branch, "
            "8784 h of 2020 at hourly and 5-minute resolution. The only "
            "public system with full UC parameters, a real network and a "
            "full year of profiles. ~290 MB."
        ),
        license_note="NREL data-use disclaimer in README; redistributable with notice",
    ),
    Dataset(
        key="nrel-118",
        directory="PowerSystemsTestData",
        url="https://github.com/Sienna-Platform/PowerSystemsTestData.git",
        summary=(
            "NREL-118: 118 buses, 327 generators (192 thermal), 186 lines "
            "with MW limits, 8784 h (leap year 2024) of day-ahead and "
            "real-time load/wind/solar. The only source with an explicit "
            "no-load heat rate ('Heat Rate Base (MMBTU/hr)'). Sparse "
            "checkout of the 118-Bus folder only."
        ),
        license_note="NO LICENSE — all rights reserved; local use only",
        sparse_paths=["118-Bus"],
    ),
]

BY_KEY = {d.key: d for d in DATASETS}


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def fetch(dataset: Dataset, force: bool = False) -> Path:
    """Clone ``dataset`` into data/external/, shallow and sparse if it can."""
    target = EXTERNAL_DIR / dataset.directory
    if target.exists():
        if not force:
            print(f"{dataset.key}: already present at {target} (use --force to re-clone)")
            return target
        shutil.rmtree(target)

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1"]
    if dataset.sparse_paths:
        # Fetch blobs on demand so the unused folders never download.
        command += ["--filter=blob:none", "--sparse"]
    command += [dataset.url, str(target)]

    print(f"{dataset.key}: cloning {dataset.url} ...")
    _run(command)
    if dataset.sparse_paths:
        _run(["git", "sparse-checkout", "set", *dataset.sparse_paths], cwd=target)
    print(f"{dataset.key}: ready at {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=[*BY_KEY, []],
        help="datasets to fetch (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="describe the datasets and exit")
    parser.add_argument("--force", action="store_true", help="re-clone even if present")
    args = parser.parse_args(argv)

    if args.list:
        for dataset in DATASETS:
            print(f"{dataset.key}  ->  data/external/{dataset.directory}")
            print(f"  {dataset.summary}")
            print(f"  license: {dataset.license_note}\n")
        return 0

    wanted = [BY_KEY[k] for k in args.datasets] if args.datasets else DATASETS
    for dataset in wanted:
        fetch(dataset, force=args.force)
        print(f"  license: {dataset.license_note}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
