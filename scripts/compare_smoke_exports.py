#!/usr/bin/env python3
"""Check two complete, two-inference smoke exports for bitwise equality."""
import argparse
import filecmp
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left = args.left / "debug_dumps"
    right = args.right / "debug_dumps"
    expected = {
        f"current_rank_{rank}_inference_{frame}_manifest.txt"
        for rank in range(24) for frame in (1, 2)
    }
    for directory in (left, right):
        actual = {path.name for path in directory.glob("*_manifest.txt")}
        if actual != expected:
            raise SystemExit(f"Incomplete inference coverage: {directory}: {len(actual)}/48 manifests")
    left_files = {path.name for path in left.glob("*.bin")}
    right_files = {path.name for path in right.glob("*.bin")}
    if not left_files or left_files != right_files:
        raise SystemExit("Binary export file sets are empty or differ")
    required = {
        f"current_rank_{rank}_inference_{frame}_{stage}.bin"
        for rank in range(24) for frame in (1, 2)
        for stage in ("raw_provider_output", "reconstructed_fields_field0",
                      "reconstructed_fields_field1", "reconstructed_fields_field2")
    }
    if not required <= left_files:
        raise SystemExit("Required inference outputs are missing")
    different = [name for name in sorted(left_files)
                 if not filecmp.cmp(left / name, right / name, shallow=False)]
    print(f"Coverage: 24 ranks x 2 inferences; binary files: {len(left_files)}")
    print(f"Bitwise equal: {len(left_files) - len(different)}; different: {len(different)}")
    for name in different[:10]:
        print(f"DIFFERENT {name}")
    if different:
        raise SystemExit(1)
    print("BITWISE_PARITY_PASS")


if __name__ == "__main__":
    main()
