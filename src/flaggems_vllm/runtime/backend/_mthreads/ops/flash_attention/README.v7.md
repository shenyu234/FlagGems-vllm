# v7 MThreads BF16 attention

The positive-scale BF16 cached D256 path computes fresh Q/K norm bounds on the
GPU. Cauchy's inequality bounds each logit by `||q|| * max ||k|| * abs(scale)`.
If every row in a CTA has bound <= 30, the fixed log2 reference is
`ceil(bound * log2(e)) + 2`. Valid probabilities stay above roughly `2^-90`,
inside BF16's normal exponent range, and at most one. This removes per-tile
max reductions and output rescaling. Larger or nonfinite bounds select the
existing stable online calculation; no input values or results are cached.

The prefill kernel uses two shared P buffers and issues current QK plus previous
PV before a wait-all. It needs no partial-wait compiler extension. Mixed query
lengths use the existing GPU compact tile mapping to avoid empty rectangular
CTAs. Tile configuration remains 128x64, 16 warps, one stage. FP16 and
nonpositive scales keep the v6 path.

The existing BF16 GQA8 D256 Short-Q Split-KV path uses the same bound to avoid
unnecessary online updates. Its norm prepass reads paged K directly; it keeps
the existing 32x32, four-warp, split policy and FP32 partial O/LSE merge.
The gate is paged long KV (capacity >=8192), positive scale, no window,
softcap or ALiBi, and Triton >=3.6. Other direct/decode paths specialize with
no norm input and retain their existing algorithm.

All norm calculation, gather, attention, and partial-output merge are included
in Graph replay timing. Updated Q/K/V, page tables and sequence lengths are
read on each replay. Host code only reads metadata and allocates empty buffers;
there is no torch compute fallback or native attention call.

`math_affine4.mu/.bc` wraps the SDK's float4 FMA and exp2 intrinsics. Like the
existing `math_packed4` helper, it is packaged beside the Python module. Rebuild:

```sh
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward
/usr/local/musa/bin/mcc -x musa --offload-arch=mp_31 --cuda-device-only \
  -fgpu-rdc -c -emit-llvm -O2 math_affine4.mu -o math_affine4.bc
```

Tests: the original MThreads/v6 suite plus
`tests/test_flash_attn_varlen_mthreads_v7.py`. The latter covers mixed fast and
online CTAs, causal alignment, empty/full-masked rows, both page sizes, and
Graph updates across data, lengths, page tables, and bound classification.
The operator remains inference-only. Offline static MThreads configuration is
intentional; NVIDIA tuning, the public API and baseline benchmark are unchanged.
