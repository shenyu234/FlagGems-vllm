# v6 MThreads prefill

The cached D256 paged-prefill route uses Triton 3.6 TLE WGMMA (MUSA SQMMA),
TME loads, packed shared P stores, float4 exp2, delayed denominator reduction,
K prefetch, and PV/softmax overlap. The tested configuration is 128x64, 16 warps,
one stage. Existing decode, Split-KV, and general routes are retained.
Triton 3.2 keeps the original descriptor implementation. No compiler patches.

`math_packed4.bc` is an mp_31 SDK math helper, not a native attention kernel.
The packaged bitcode is located relative to this module; experiment directories
and PYTHONPATH overrides for math libraries are not required. Rebuild with:

```sh
/usr/local/musa/bin/mcc -x musa --offload-arch=mp_31 --cuda-device-only -fgpu-rdc -c -emit-llvm -O2 math_packed4.mu -o math_packed4.bc
```

The route inherits the existing contiguous input/output and 256 MiB gather
workspace gates. Nonpositive softmax scales disable raw-score maximum tracking.
GPU sequence lengths, KV data, and page tables are re-read on every graph replay.
This is an inference-only specialization with a fixed, previously tuned
configuration; NVIDIA tuning and public exports are unchanged.
