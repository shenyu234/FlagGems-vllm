"""Single-file pytest plugin for strict CUDA/MUSA Graph benchmark timing.

Load with ``python -m pytest -p _cudagraph_plugin``. Both Triton timing
entry points use GPU graph replay; capture failures propagate, never silently
fall back to eager timing. No modifications to FlagGems are needed.

Only GPU work captured from fn is replayed. Compilation, warmup, and cached
one-time weight repacking are excluded. GPU repacking executed during capture
is replayed and timed on every iteration. This plugin does not check accuracy.
"""

from __future__ import annotations

import functools
import inspect
import math
import os
import sys
import time

import pytest
import torch
import triton.testing as _tt

_ORIGINAL_DO_BENCH = _tt.do_bench
_ORIGINAL_DO_BENCH_CUDAGRAPH = _tt.do_bench_cudagraph
_DIRECT_SIGNATURE = inspect.signature(_ORIGINAL_DO_BENCH)
_GRAPH_SIGNATURE = inspect.signature(_ORIGINAL_DO_BENCH_CUDAGRAPH)
_CONFIGURED_WARMUP_MS = 25.0
_MIN_WARMUP_ITERS = 5
_MAX_WARMUP_ITERS = 200


def _color_enabled():
    mode = os.environ.get("FLAGGEMS_PERF_COLOR", "").strip().lower()
    if mode in ("always", "1", "yes", "on"):
        return True
    if mode in ("never", "0", "no", "off"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force.strip().lower() not in ("", "0", "false", "no", "off", "none")
    return not os.environ.get("NO_COLOR") and sys.stderr.isatty()


def _log(message, *, failed=False):
    if _color_enabled():
        color = "\033[31m" if failed else "\033[94m"
        message = f"{color}{message}\033[0m"
    print(message, file=sys.stderr, flush=True)


def _device_backend(device_type=None):
    musa = getattr(torch, "musa", None)
    if device_type is None:
        device_type = "musa" if musa is not None and musa.is_available() else "cuda"
    if device_type not in ("cuda", "musa"):
        raise ValueError("_cudagraph_plugin supports CUDA and MUSA devices only")
    device = getattr(torch, device_type, None)
    if device is None or not device.is_available():
        raise RuntimeError(f"_cudagraph_plugin requires an available {device_type} device")
    return device_type, device


def _warmup_before_capture(fn, warmup_ms, grad_to_none, device):
    """Bounded side-stream warmup; warmup_ms is a target, not a guarantee.

    The first call absorbs lazy initialization/JIT/autotune/repacking. A timed
    eager call then estimates a warmup count capped at 5..200 iterations.
    These warmup calls are outside the reported graph replay latency.
    """
    def run():
        if grad_to_none is not None:
            for tensor in grad_to_none:
                tensor.grad = None
        fn()

    # Inputs may have been produced on another stream. Complete their work
    # before warmup and Triton's own side-stream capture/measurement.
    device.synchronize()
    stream = device.Stream()
    with device.stream(stream):
        run()
        stream.synchronize()
        start = time.perf_counter()
        run()
        stream.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        count = math.ceil(warmup_ms / max(elapsed_ms, 1e-6))
        count = max(_MIN_WARMUP_ITERS, min(count, _MAX_WARMUP_ITERS))
        for _ in range(count):
            run()
    device.synchronize()


def _do_bench_musagraph(
    fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean", n_retries=10
):
    """Triton's graph timing procedure using MUSA's stream, graph and events."""
    if return_mode not in ("min", "max", "mean", "median", "all"):
        raise ValueError(f"Unsupported return_mode: {return_mode}")
    rep = float(rep)
    if not math.isfinite(rep) or rep <= 0:
        raise ValueError("rep must be finite and positive")
    if not isinstance(n_retries, int) or n_retries < 1:
        raise ValueError("n_retries must be a positive integer")
    device = torch.musa

    def run():
        if grad_to_none is not None:
            for tensor in grad_to_none:
                tensor.grad = None
        fn()

    stream = device.Stream()
    with device.stream(stream):
        run()
        start = device.Event(enable_timing=True)
        end = device.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            run()
        end.record()
        device.synchronize()
        estimate_ms = start.elapsed_time(end) / 5
        n_repeat = max(1, int(rep / estimate_ms)) if estimate_ms > 0 else 1000

        graph = device.MUSAGraph()
        with device.graph(graph, stream=stream):
            for _ in range(n_repeat):
                run()
        device.synchronize()

        times = []
        for _ in range(n_retries):
            start = device.Event(enable_timing=True)
            end = device.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            device.synchronize()
            times.append(start.elapsed_time(end) / n_repeat)

    # Match installed Triton's statistics on CPU, outside the timed interval.
    return _tt._summarize_statistics(
        torch.tensor(times, dtype=torch.float32, device="cpu"), quantiles, return_mode
    )


def _measure_graph(fn, warmup_ms, graph_kwargs, device_type=None):
    warmup_ms = float(warmup_ms)
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("warmup must be finite and nonnegative")
    # Validate the installed Triton API before running or capturing any work.
    _GRAPH_SIGNATURE.bind(fn, **graph_kwargs)
    op = os.environ.get("FLAGGEMS_PERF_CURRENT_OP", getattr(fn, "__name__", "unknown"))
    phase = "device"
    try:
        backend, device = _device_backend(device_type)
        phase = "warmup"
        _warmup_before_capture(fn, warmup_ms, graph_kwargs.get("grad_to_none"), device)
        phase = "capture_or_replay"
        benchmark = (
            _do_bench_musagraph if backend == "musa" else _ORIGINAL_DO_BENCH_CUDAGRAPH
        )
        result = benchmark(fn, **graph_kwargs)
    except Exception as exc:
        detail = " ".join(str(exc).split())[:240]
        _log(
            f"BENCHMARK_CUDAGRAPH_FAILED op={op} phase={phase} "
            f"reason={type(exc).__name__} detail={detail!r}; no eager fallback",
            failed=True,
        )
        raise
    _log(f"BENCHMARK_CUDAGRAPH_OK op={op} timing=cudagraph backend={backend}")
    return result


@functools.wraps(_ORIGINAL_DO_BENCH)
def _patched_do_bench(fn, *args, **kwargs):
    bound = _DIRECT_SIGNATURE.bind(fn, *args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    # Deliberately replace eager timing (including its L2-cache behavior).
    graph_kwargs = {
        name: values[name]
        for name in ("rep", "grad_to_none", "quantiles", "return_mode")
        if name in values
    }
    return _measure_graph(
        fn, values.get("warmup", 25), graph_kwargs, values.get("device_type")
    )


@functools.wraps(_ORIGINAL_DO_BENCH_CUDAGRAPH)
def _patched_do_bench_cudagraph(fn, *args, **kwargs):
    bound = _GRAPH_SIGNATURE.bind(fn, *args, **kwargs)
    bound.apply_defaults()
    # Preserve version-specific options such as Flagtree's n_retries.
    graph_kwargs = dict(bound.arguments)
    graph_kwargs.pop("fn")
    return _measure_graph(fn, _CONFIGURED_WARMUP_MS, graph_kwargs)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    global _CONFIGURED_WARMUP_MS
    _CONFIGURED_WARMUP_MS = float(config.getoption("--warmup", default=25))
    _tt.do_bench = _patched_do_bench
    _tt.do_bench_cudagraph = _patched_do_bench_cudagraph
    _log("[cudagraph-plugin] strict CUDA/MUSA Graph timing enabled for both Triton APIs")


def pytest_unconfigure(config):
    if _tt.do_bench is _patched_do_bench:
        _tt.do_bench = _ORIGINAL_DO_BENCH
    if _tt.do_bench_cudagraph is _patched_do_bench_cudagraph:
        _tt.do_bench_cudagraph = _ORIGINAL_DO_BENCH_CUDAGRAPH
