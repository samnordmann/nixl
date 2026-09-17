# nixl

This is a *meta package*. The PyPI distribution installs both the CUDA 12
and CUDA 13 backends, and the correct one is selected automatically at
runtime based on the CUDA version reported by PyTorch. Source builds install
a single backend unless built with `-Drelease_wheel=true`.

```bash
pip install nixl
```

The `nixl[cu12]` and `nixl[cu13]` extras are accepted for backwards
compatibility but have no additional effect.

CuTe DSL is optional. On CUDA 12, install the exact version qualified with this
NIXL release through the explicit meta-package extra:

```bash
pip install 'nixl[cute-cu12]'
```

CUDA 13 intentionally has no CuTe extra in this release. CUTLASS DSL 4.5.1's
official `cu13` dependency set has an
[upstream overlapping-wheel issue](https://github.com/NVIDIA/cutlass/issues/3259),
and its plain/base payload is not interchangeable. Use CUTLASS's source
`setup.sh --cu13` workflow in an independently verified environment until a
fixed DSL release is qualified. `nixl.device.cute` still verifies exact
`nvidia-cutlass-dsl==4.5.1` distribution metadata at import. The CUDA 12 extra
also constrains `cuda-python>=12.8,<13` so dependency resolution cannot mix a
CUDA 13 binding into a CUDA 12 backend. Source-only
environments lacking that metadata must explicitly set
`NIXL_CUTE_ALLOW_UNQUALIFIED_CUTLASS_DSL=1`; the bypass emits a warning and is
not a release-qualified configuration.
