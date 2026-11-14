"""Split a large CSV into equally sized parts, repeating the header row.

Usage:
    py scripts/split_csv.py <input.csv> [--parts 4] [--out-dir DIR]

Parts are written next to the input as <stem>_part1.csv, <stem>_part2.csv, ...
Splitting is done with the csv module, so quoted fields containing commas or
newlines are handled correctly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def split_csv(path: Path, parts: int, out_dir: Path) -> list[Path]:
    if parts < 2:
        raise ValueError("--parts must be >= 2")
    if not path.is_file():
        raise FileNotFoundError(path)

    out_dir.mkdir(parents=True, exist_ok=True)

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty") from None

        data_rows = sum(1 for _ in reader)

    rows_per_part = -(-data_rows // parts)
    if rows_per_part == 0:
        raise ValueError(f"{path} has no data rows")

    outputs = [out_dir / f"{path.stem}_part{i + 1}.csv" for i in range(parts)]

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)

        handles = [
            out.open("w", encoding="utf-8", newline="") for out in outputs
        ]
        try:
            writers = [csv.writer(h) for h in handles]
            for writer in writers:
                writer.writerow(header)

            for index, row in enumerate(reader):
                part = min(index // rows_per_part, parts - 1)
                if len(row) != len(header):
                    raise ValueError(
                        f"Row {index + 2} has {len(row)} fields, expected "
                        f"{len(header)}; aborting"
                    )
                writers[part].writerow(row)
        finally:
            for handle in handles:
                handle.close()

    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--parts", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = args.out_dir or args.input.parent
    outputs = split_csv(args.input, args.parts, out_dir)

    for out in outputs:
        size_mib = out.stat().st_size / (1024 * 1024)
        print(f"{out}  ({size_mib:.1f} MiB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
