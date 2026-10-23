"""Build the block lookup CSV by delegating to the GPU block profiler.

The lookup is produced in a single pass by `benchmarking.profile_gpu_blocks`,
which profiles every block in the input CSV, then synthesises and profiles the
FC head (block_idx=5: the projection conv and the classifier). This module is a
thin front-end over that, so the lookup has one point of truth.

Usage:
    python -m scoring.build_block_lookup \\
        --input-csv data/block_lookup/idw_block_lookup.csv \\
        --output-csv data/block_lookup/idw_block_lookup_jetson_gpu.csv \\
        --device cuda:0 --skip-existing

Any further flags are passed straight through to the profiler
(`--warmup`, `--runs`, `--no-fc`, `--fc-num-classes`, ...). Requires a CUDA
device, and latency is device-specific: profiling on different hardware
produces a different lookup.
"""

from __future__ import annotations

import argparse
import sys

from utils.paths import LOOKUP_CSV, setup_model_path, setup_repo_path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Build the block lookup CSV via benchmarking.profile_gpu_blocks."
    )
    parser.add_argument("--input-csv", "--input_csv", dest="input_csv", required=True,
                        help="Block grid CSV holding block_str/channels/flops/params/scores")
    parser.add_argument("--output-csv", "--output_csv", dest="output_csv",
                        default=str(LOOKUP_CSV),
                        help="Where to write the lookup with latency + FC rows")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    setup_repo_path()
    setup_model_path()

    from benchmarking import profile_gpu_blocks

    sys.argv = [
        "profile_gpu_blocks",
        "--input-csv", args.input_csv,
        "--output-csv", args.output_csv,
        *passthrough,
    ]
    profile_gpu_blocks.main()


if __name__ == "__main__":
    main()
