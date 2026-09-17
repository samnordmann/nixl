# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the example's private device shims with the same UCX as host NIXL."""

import argparse
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


def clean_ir(text):
    # libNVVM rejects Clang's FTZ module flag and explicit function alignment.
    ids = re.findall(r"^(!\d+)\s*=.*nvvm-reflect-ftz.*$", text, re.MULTILINE)
    lines = []
    for line in text.splitlines():
        if any(line.startswith(f"{key} =") for key in ids):
            continue
        if line.startswith("!llvm.module.flags ="):
            for key in ids:
                line = re.sub(rf"\s*{re.escape(key)}(?!\d),?", "", line)
            line = re.sub(r"!\{\s*,?\s*", "!{", line)
            line = re.sub(r",\s*}", "}", line)
        if line.startswith("define "):
            line = re.sub(r"\s+align\s+\d+(?=\s|\{)", "", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucx", required=True, type=Path)
    parser.add_argument("--arch", required=True, help="e.g. sm_90 or sm_100")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--llvm", type=Path, default=Path("/usr/lib/llvm-20/bin"))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    includes = [
        here / "compat",
        root / "src/api/device",
        root / "src/api/cpp",
        args.ucx / "include",
        args.cuda / "include/cccl",
    ]
    doca = subprocess.run(
        ["pkg-config", "--cflags", "doca-gpunetio"], text=True, capture_output=True
    )
    extra = shlex.split(doca.stdout) if doca.returncode == 0 else []

    def run(tool, *flags):
        subprocess.run([str(args.llvm / tool), *map(str, flags)], check=True)

    with tempfile.TemporaryDirectory() as tmp:
        raw, optimized, cleaned = (
            Path(tmp) / name for name in ("raw.bc", "opt.ll", "clean.ll")
        )
        run(
            "clang++",
            "-std=c++17",
            "-x",
            "cuda",
            "--cuda-device-only",
            f"--cuda-path={args.cuda}",
            f"--cuda-gpu-arch={args.arch}",
            "-Wno-unknown-cuda-version",
            "-D_NV_RSQRT_SPECIFIER=",
            "-DNDEBUG",
            "-O1",
            "-emit-llvm",
            "-c",
            *(f"-I{path}" for path in includes),
            *extra,
            here / "device.cu",
            "-o",
            raw,
        )
        run(
            "opt",
            "--passes=internalize,inline,globaldce",
            "-internalize-public-api-list=cute_nixl_put,cute_nixl_signal,cute_nixl_wait",
            "-S",
            raw,
            "-o",
            optimized,
        )
        cleaned.write_text(clean_ir(optimized.read_text()))
        run("llvm-as", cleaned, "-o", here / "device.bc")
    print(
        f"Built {here / 'device.bc'} for {args.arch}; rebuild after changing NIXL/UCX/GPU."
    )


if __name__ == "__main__":
    main()
