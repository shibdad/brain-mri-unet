"""Download and extract the LGG MRI segmentation dataset from Kaggle.

Prerequisites
-------------
1. A Kaggle account and an API token. Create one at
   https://www.kaggle.com/settings -> "Create New API Token" (downloads
   kaggle.json).
2. Place kaggle.json at ~/.kaggle/kaggle.json  (or set KAGGLE_USERNAME /
   KAGGLE_KEY env vars). On Colab, upload it in the notebook instead.

Usage
-----
    python scripts/download_data.py --out data

This downloads `mateuszbuda/lgg-mri-segmentation` (~750 MB) and extracts it to
`data/lgg-mri-segmentation/`, which then contains the `kaggle_3m/` folder that
`src.data` expects.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

DATASET = "mateuszbuda/lgg-mri-segmentation"


def main() -> None:
    p = argparse.ArgumentParser(description="Download LGG MRI dataset from Kaggle.")
    p.add_argument("--out", default="data", help="Output directory (default: data).")
    args = p.parse_args()

    target = os.path.join(args.out, "lgg-mri-segmentation")
    os.makedirs(target, exist_ok=True)

    try:
        import kaggle  # noqa: F401  (import validates credentials are present)
    except OSError as e:
        sys.exit(
            f"Kaggle credentials not found: {e}\n"
            "Place kaggle.json at ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME / KAGGLE_KEY."
        )
    except ImportError:
        sys.exit("The 'kaggle' package is not installed. Run: pip install kaggle")

    print(f"Downloading {DATASET} -> {target} (this can take a few minutes)...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET,
            "-p", target,
            "--unzip",
        ],
        check=True,
    )
    print("Done. Point training at this path, e.g.:")
    print(f"    python -m src.train --data-dir {target} --epochs 50 --out runs/exp1")


if __name__ == "__main__":
    main()
