#!/usr/bin/env python3
"""Print data/label shapes and total pixel counts for a given dataset."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Load_Data import load_data


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset dimensions")
    parser.add_argument("dataset", help="Dataset key, e.g. SF, FL_T, ober")
    args = parser.parse_args()

    data, gt = load_data(args.dataset)
    print(f"dataset: {args.dataset}")
    print(f"data shape: {data.shape}")
    print(f"gt shape: {gt.shape}")
    total_pixels = gt.shape[0] * gt.shape[1]
    print(f"total pixels: {total_pixels}")


if __name__ == "__main__":
    main()
