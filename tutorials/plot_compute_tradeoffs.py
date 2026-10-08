"""
Compute time and memory footprint trade-offs
============================================

This gallery explores how TabICL's computational parameters affect
compute time and memory footprints (host and device memory), covering
device choice and use of the KV cache.

.. note::
   The figures in the gallery are generated in the
   `tabicl-benchmarks <https://github.com/probabl/tabicl-benchmarks>`_
   project repository. The version of ``tabicl`` displayed in figure footnotes
   may not match the latest ``tabicl`` release.

.. raw:: html

   <style>.sphx-glr-timing{display:none !important}</style>
"""

# %%
# Introducing kv-caching
# ----------------------
#
# At the core of ``tabicl``'s architecture are transformer-like components, and
# much like other transformer-based systems, it can benefit from temporarily
# storing intermediate objects computed on the way to the final output, so
# they can be reused later to speed up predictions on similar inputs, or even,
# in some cases, to reduce the memory footprint.
#
# These objects are called, in transformer terminology, *keys* and *values*,
# hence the name of the memoization system: the *kv*-cache.
#
# Tabular foundational models encode the *entire* training dataset along with
# the training labels. In this case, enabling the key-value cache means that
# all the operations on the training data that do not depend on the target
# dataset are stored and available for reuse to speed up prediction on
# previously unseen target datasets.
#
# By default, the :meth:`.fit <tabicl.TabICLClassifier.fit>` method of
# ``tabicl`` estimators does not run any compute-heavy operations. It merely
# downloads (if not already present) and loads the model weights, and stores the
# training data after applying some light pre-processing. At predict time,
# both the training data and the input dataset are jointly forwarded to the
# actual compute machinery.
#
# With kv-cache enabled, :meth:`.fit <tabicl.TabICLClassifier.fit>` additionally
# computes some of the intermediate outputs that only depend on the training
# data, and stores them in host memory if ``device='cpu'``, otherwise on device
# memory.
#
# The ``tabicl`` API exposes several caching levels:
#
# - ``kv_cache=False`` (the default) to disable caching
# - ``kv_cache=True`` or ``kv_cache='kv'`` for aggressive caching
# - ``kv_cache='repr'`` for a moderate amount of caching, tuned down in the
#   later layers
#
# The :doc:`Getting Started <getting_started>` gallery already showcases usage
# of this parameter. The present gallery provides an in-depth analysis of its
# effects on a range of workloads.

# %%
# Effects of ``kv_cache`` on device compute time
# -----------------------------------------------
#
# The figure plots predict time and total fit+predict time against three
# dataset-size axes: number of training rows, number of features, and size of
# test data, with and without kv-caching.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig1.html
#
# As expected, enabling kv-caching considerably speeds up the subsequent
# :meth:`.predict <tabicl.TabICLClassifier.predict>` calls. When it is
# enabled, :meth:`.fit <tabicl.TabICLClassifier.fit>` absorbs most of the
# operations related to the training data, so that
# :meth:`.predict <tabicl.TabICLClassifier.predict>` compute time no longer
# depends on the number of training rows (see left plot). Naturally, since
# kv-caching only affects operations on training data, the effect on total
# :meth:`.predict <tabicl.TabICLClassifier.predict>` time is less noticeable
# with increased test-data size, and the speed-up eventually vanishes for very
# large test data (see right plot).
#
# The fit+predict curves, where a :meth:`.fit <tabicl.TabICLClassifier.fit>`
# call is followed by a single subsequent
# :meth:`.predict <tabicl.TabICLClassifier.predict>` call, are nearly
# identical between the two modes. This reveals that enabling kv-caching
# moves some of the time budget from
# :meth:`.predict <tabicl.TabICLClassifier.predict>` to
# :meth:`.fit <tabicl.TabICLClassifier.fit>`, without any noticeable overhead.
# The effect on :meth:`.predict <tabicl.TabICLClassifier.predict>` speed is
# even more beneficial for large training data (see left plot).
#
# Furthermore, the predict and fit+predict curves converge for large training
# data (without kv-caching) or for large test data. This trend highlights the
# computational complexity of ``tabicl`` in different regimes. The complexity
# of loading the model weights and preprocessing the data scales linearly with
# the amount of data. On the other hand, computation of the intermediate hidden
# activations within the transformer scales quadratically. For large amounts
# of data, the former becomes negligible compared to the latter, but it remains
# a significant offset for smaller data.
#
# Finally, the middle plot shows that the effect of the cache on
# :meth:`.predict <tabicl.TabICLClassifier.predict>` time grows super-linearly
# as the number of features increases. This means kv-caching is especially
# beneficial for wide tables.
#
# In a nutshell, enabling ``kv-caching`` has **no downside** on compute time;
# the effect is negligible at worst. The trade-off is a higher memory
# footprint, which the next section quantifies.

# %%
# Effects of ``kv_cache`` on memory
# ---------------------------------
#
# The previous figure displays the compute-time side of the KV-cache trade-off;
# this one displays the memory side. Peak memory usage is recorded separately
# for the fit and predict phases. The peak memory allocated during :meth:`.fit
# <tabicl.TabICLClassifier.fit>` when kv-caching is disabled is always
# negligible, and is not displayed. When kv-caching is enabled, the peak memory
# reported during :meth:`.predict <tabicl.TabICLClassifier.predict>` includes
# the cache overhead.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig2.html
# 
# Somewhat surprisingly, the memory footprint of
# :meth:`.predict <tabicl.TabICLClassifier.predict>` is uniformly lower
# when the key-value cache is enabled, so caching also appears to have **no
# downside** with respect to memory usage. In fact, the
# :meth:`.predict <tabicl.TabICLClassifier.predict>` memory curves are
# remarkably similar in shape to the compute-time plots, and the same
# conclusions apply. The reason is that the overhead added by the cache is 
# more than offset by the memory saved by skipping the computation of intermediate 
# hidden activations.
#
# The left and right plots, however, suggest that for very large training data
# (or very small test data), the peak memory needed to build the key-value
# cache (during :meth:`.fit <tabicl.TabICLClassifier.fit>`) surpasses the peak
# memory of :meth:`.predict <tabicl.TabICLClassifier.predict>` without the
# cache. In those cases, there may be a genuine trade-off: caching is
# worthwhile only if enough memory is available to build the cache.
#
# The particularly high peak memory observed during
# :meth:`.fit <tabicl.TabICLClassifier.fit>` is the sum of the peak memory
# needed to compute the intermediate activations in the later layers, and the
# memory held by the objects from earlier layers that are retained in the cache
# being built.

# %%
# Practical summary
# -----------------
#
# In short, a good rule overall is to *always activate the kv-cache* unless the
# available memory is too limited to build the cache.

# %%
# Other limitations
# -----------------
#
# One important limitation is that caching is not implemented for
# classification problems with more than 10 classes. In this case, the
# :meth:`.fit <tabicl.TabICLClassifier.fit>` call raises an exception instead.

# sphinx_gallery_start_ignore
# -----------------------------------------------------------------------------
# Benchmark code used to (re)generate the figures of this gallery.
#
# The documentation build embeds the pre-rendered figures from
# ``docs/_static/compute_tradeoffs``; the code below is hidden from the
# rendered page and skipped when executed by sphinx-gallery. Run this file
# directly to re-run the benchmarks on the current branch and hardware:
#
#     python tutorials/plot_compute_tradeoffs.py --help
#     python tutorials/plot_compute_tradeoffs.py --output-dir out/ --n-repeats 3
#     python tutorials/plot_compute_tradeoffs.py --quick --figures 1 2  # smoke test
#
# Each figure ``N`` is written to ``<output-dir>/figN.html`` (a drop-in
# replacement for ``docs/_static/compute_tradeoffs/figN.html``, containing a
# desktop and a mobile layout) and ``<output-dir>/figN_preview.html`` (a
# standalone page with the desktop layout only). Raw measurements and run
# metadata (tabicl version, git branch/commit, device, ...) are written to
# ``<output-dir>/results.json`` for comparison across branches and hardware.
#
# Measurement protocol:
#
# - Synthetic 10-class classification data (``make_classification``), with
#   ``TabICLClassifier(n_estimators=4)``.
# - Each (data size, kv_cache, device, offload_mode) configuration (and
#   repeat) runs with a fresh estimator in a fresh Python subprocess, after a
#   small warm-up fit/predict in that process, so that one-off costs (backend
#   initialization, first checkpoint read from disk) and memory left over by
#   previous runs do not leak into the measurements (``--no-isolate`` runs
#   everything in-process instead: faster, but host memory gets unreliable).
# - ``fit`` time includes loading the checkpoint (as in regular usage);
#   "fit + predict" is the sum of a ``fit`` and a single subsequent
#   ``predict``.
# - Device memory: peak memory allocated by torch on the accelerator during
#   each phase (``torch.<backend>.max_memory_allocated`` after resetting the
#   peak stats), i.e. it includes model weights and any KV cache that is
#   resident on the device. On MPS, the allocated memory is sampled instead.
# - Host memory (RAM): peak process RSS during each phase, sampled every few
#   milliseconds, minus the RSS measured right before the estimator was
#   created (so it includes the model weights when they live in RAM).
# - With ``device='cpu'``, figure 2 shows host memory instead of device memory.
# - With ``--n-repeats > 1``, the median over repeats is reported.
# - Configurations that run out of memory are reported as missing points.
# -----------------------------------------------------------------------------
import argparse
import ctypes
import gc
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Sphinx-gallery executes this file while building the docs: skip the
# benchmarks there (the docs embed pre-rendered figures) unless explicitly
# requested through the environment.
RUN_BENCHMARKS = "sphinx_gallery" not in sys.modules or os.environ.get("TABICL_RUN_COMPUTE_BENCHMARKS", "0") == "1"

# Benchmark grids, matching the published figures.
N_CLASSES = 10
N_ESTIMATORS = 4
TRAIN_ROWS_SWEEP = dict(values=[300, 1000, 2000, 4000, 8000, 16000], n_features=100, n_test=2000)
FEATURES_SWEEP = dict(values=[20, 40, 100, 200, 500], n_train=1000, n_test=2000)
TEST_ROWS_SWEEP = dict(values=[2, 30, 500, 2000, 8000], n_train=4000, n_features=100)
OFFLOAD_CONFIG = dict(n_train=4000, n_features=200, n_test_small=200, n_test_large=10_000)

# Much smaller grids for a quick smoke test of the script (``--quick``).
QUICK_TRAIN_ROWS_SWEEP = dict(values=[100, 300, 1000], n_features=20, n_test=200)
QUICK_FEATURES_SWEEP = dict(values=[10, 20, 40], n_train=300, n_test=200)
QUICK_TEST_ROWS_SWEEP = dict(values=[2, 30, 200], n_train=300, n_features=20)
QUICK_OFFLOAD_CONFIG = dict(n_train=300, n_features=20, n_test_small=50, n_test_large=1000)

# Plot styling, matching the published figures.
COLORS = {"no_cache": "#009E73", "cache": "#D55E00", "time": "#000000", "ram": "#0072B2", "vram": "#D55E00"}
SYMBOLS = {"no_cache": "circle", "cache": "square"}
CACHE_LABELS = {"no_cache": "no kv cache", "cache": "kv cache"}
TITLE_FONT = dict(size=13, color="#1f2937")
TICK_FONT = dict(size=11, color="#374151")
GRID_COLOR = "rgba(0,0,0,0.10)"
SUBTITLE_STYLE = "font-size:13px;color:#6b7280"
FOOTNOTE_STYLE = "font-size:10px;color:#9ca3af"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-run the compute time / memory benchmarks of the compute trade-offs gallery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("compute_tradeoffs_output"),
        help="Directory where figures and raw results are written (default: %(default)s).",
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Figures to generate (default: all).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used for figures 1-2 and as accelerator in figure 3 "
        "(default: tabicl's default device, i.e. cuda > xpu > mps > cpu).",
    )
    parser.add_argument(
        "--offload-mode",
        default="auto",
        choices=["auto", "gpu", "cpu", "disk"],
        help="offload_mode used for figures 1-2 (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local checkpoint passed as model_path (default: download/cache the default checkpoint).",
    )
    parser.add_argument("--n-repeats", type=int, default=1, help="Repeats per configuration (default: %(default)s).")
    parser.add_argument(
        "--skip-cpu-baseline",
        action="store_true",
        help="Do not benchmark device='cpu' in figure 3 when an accelerator is used (it is slow).",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Run all configurations in this process instead of one fresh subprocess per run. "
        "Faster, but host memory measurements become unreliable.",
    )
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)  # internal: run one configuration
    parser.add_argument("--quick", action="store_true", help="Use small grids to quickly smoke-test the script.")
    parser.add_argument("--show", action="store_true", help="Also open the (desktop) figures in a browser.")
    # parse_known_args: tolerate extra arguments, e.g. when run from an IPython kernel.
    args, _ = parser.parse_known_args(argv)
    return args


# --- Environment / metadata --------------------------------------------------


def _cpu_name():
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def device_label(device):
    """Human-readable hardware description, e.g. ``GPU (NVIDIA L4)``."""
    import torch

    if device.type == "cuda":
        return f"GPU ({torch.cuda.get_device_name(device)})"
    if device.type == "xpu":
        return f"XPU ({torch.xpu.get_device_name(device)})"
    if device.type == "mps":
        return f"MPS ({_cpu_name()})"
    if device.type == "cpu":
        return f"CPU ({_cpu_name()}, {torch.get_num_threads()} threads)"
    return device.type.upper()


def device_short_label(device):
    return {"cuda": "GPU", "cpu": "CPU"}.get(device.type, device.type.upper())


def _git_info(path):
    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(path), *cmd], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        info = {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "--short", "HEAD")}
        info["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
        return info
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(device):
    import sklearn
    import torch

    import tabicl
    from tabicl.__about__ import __version__

    return {
        "tabicl_version": __version__,
        "tabicl_path": str(Path(tabicl.__file__).parent),
        "git": _git_info(Path(tabicl.__file__).parent),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "device_label": device_label(device),
        "cpu": _cpu_name(),
        "torch_num_threads": torch.get_num_threads(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def footnote_text(metadata):
    text = f"TabICL {metadata['tabicl_version']}"
    git = metadata.get("git")
    if git:
        text += f" ({git['branch']}@{git['commit']}{', dirty' if git['dirty'] else ''})"
    return text


# --- Measurements ------------------------------------------------------------


def _backend_api(device):
    import torch

    return None if device.type == "cpu" else getattr(torch, device.type, None)


def _synchronize(device):
    api = _backend_api(device)
    if api is not None and callable(getattr(api, "synchronize", None)):
        api.synchronize()


def _empty_cache(device):
    gc.collect()
    if sys.platform.startswith("linux"):
        # Return memory freed by glibc's allocator to the OS, so it does not
        # hide in the RSS baseline of the next measurement.
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
    api = _backend_api(device)
    if api is not None and callable(getattr(api, "empty_cache", None)):
        api.empty_cache()


class _PeakMemorySampler:
    """Track peak process RSS and (when torch has no peak stats) device memory."""

    def __init__(self, device, interval=0.002):
        import psutil

        self._process = psutil.Process()
        self._interval = interval
        api = _backend_api(device)
        self._has_peak_stats = api is not None and callable(getattr(api, "max_memory_allocated", None))
        self._api = api
        self._sample_device = (
            api is not None and not self._has_peak_stats and callable(getattr(api, "current_allocated_memory", None))
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)
        if self._sample_device:
            self.peak_device = max(self.peak_device, self._api.current_allocated_memory())

    def _run(self):
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self):
        self.peak_rss = 0
        self.peak_device = 0
        if self._has_peak_stats:
            self._api.reset_peak_memory_stats()
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()
        if self._has_peak_stats:
            self.peak_device = self._api.max_memory_allocated()
        return False

    @property
    def device_mb(self):
        if self._api is None or not (self._has_peak_stats or self._sample_device):
            return None
        return self.peak_device / 2**20


def _is_oom(exc):
    import torch

    oom_types = tuple(t for t in (getattr(torch, "OutOfMemoryError", None), MemoryError) if t is not None)
    return isinstance(exc, oom_types) or "out of memory" in str(exc).lower()


def make_data(n_train, n_features, n_test, seed=0):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=n_features,
        n_informative=min(n_features, 10),
        n_redundant=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=seed,
    )
    X = X.astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:]


def _measure_phase(fn, device, baseline_rss):
    with _PeakMemorySampler(device) as sampler:
        tic = time.perf_counter()
        fn()
        _synchronize(device)
        elapsed = time.perf_counter() - tic
    return {
        "time_s": elapsed,
        "ram_mb": max(sampler.peak_rss - baseline_rss, 0) / 2**20,
        "device_mb": sampler.device_mb,
    }


def run_once(n_train, n_features, n_test, kv_cache, device, offload_mode, model_path=None):
    """Fit and predict once with a fresh estimator; return timings and peak memory."""
    import psutil

    from tabicl import TabICLClassifier

    X_train, y_train, X_test = make_data(n_train, n_features, n_test)
    with tempfile.TemporaryDirectory(prefix="tabicl_offload_") as offload_dir:
        _empty_cache(device)
        baseline_rss = psutil.Process().memory_info().rss
        clf = TabICLClassifier(
            n_estimators=N_ESTIMATORS,
            kv_cache=kv_cache,
            device=device,
            offload_mode=offload_mode,
            disk_offload_dir=offload_dir,
            model_path=model_path,
        )
        try:
            fit = _measure_phase(lambda: clf.fit(X_train, y_train), device, baseline_rss)
            predict = _measure_phase(lambda: clf.predict(X_test), device, baseline_rss)
        finally:
            clf = None  # release the model (and kv cache) before the next run
            _empty_cache(device)
    return {"fit": fit, "predict": predict}


WORKER_RESULT_PREFIX = "__TABICL_BENCHMARK_RESULT__"
WARMUP_SIZES = dict(n_train=100, n_features=10, n_test=10)


def run_worker(config_json):
    """Entry point of the subprocess running a single isolated configuration."""
    from tabicl._torch_devices import resolve_torch_device

    config = json.loads(config_json)
    config["device"] = resolve_torch_device(config["device"])
    try:
        run_once(**{**config, **WARMUP_SIZES, "kv_cache": False})
        result = run_once(**config)
    except Exception as exc:
        if not _is_oom(exc):
            raise
        result = {"oom": f"{type(exc).__name__}: {exc}"}
    print(WORKER_RESULT_PREFIX + json.dumps(result), flush=True)


def run_isolated(**config):
    """Run :func:`run_once` with a warm-up in a fresh Python process.

    A fresh process gives a clean baseline for the host memory measurements
    (allocators keep memory freed by previous runs) and for device memory.
    """
    config = {**config, "device": str(config["device"])}
    if config.get("model_path") is not None:
        config["model_path"] = str(config["model_path"])
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(config)],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith(WORKER_RESULT_PREFIX)]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"Benchmark worker failed (exit code {proc.returncode}) for {config}:\n{proc.stdout}")
    result = json.loads(lines[-1][len(WORKER_RESULT_PREFIX) :])
    if "oom" in result:
        raise MemoryError(result["oom"])
    return result


class Benchmark:
    """Run (and memoize) benchmark configurations."""

    def __init__(self, n_repeats=1, model_path=None, isolate=True):
        self.n_repeats = n_repeats
        self.model_path = model_path
        # Subprocesses need a script to run: not available e.g. in notebooks.
        self.isolate = isolate and "__file__" in globals()
        self.records = {}
        self._warmed_up = set()

    def _run_in_process(self, **config):
        key = (str(config["device"]), config["offload_mode"])
        if key not in self._warmed_up:
            run_once(**{**config, **WARMUP_SIZES, "kv_cache": False})
            self._warmed_up.add(key)
        return run_once(**config)

    def run(self, n_train, n_features, n_test, kv_cache, device, offload_mode):
        key = (n_train, n_features, n_test, bool(kv_cache), str(device), offload_mode)
        if key in self.records:
            return self.records[key]["result"]
        config = dict(
            n_train=n_train,
            n_features=n_features,
            n_test=n_test,
            kv_cache=bool(kv_cache),
            device=device,
            offload_mode=offload_mode,
            model_path=self.model_path,
        )
        print(
            f"n_train={n_train:>6} n_features={n_features:>4} n_test={n_test:>6} "
            f"kv_cache={bool(kv_cache)!s:<5} device={device!s:<6} offload={offload_mode:<4}",
            end=" ... ",
            flush=True,
        )
        repeats = []
        try:
            for _ in range(self.n_repeats):
                repeats.append(run_isolated(**config) if self.isolate else self._run_in_process(**config))
        except Exception as exc:
            if not _is_oom(exc):
                raise
            print(f"out of memory ({type(exc).__name__})")
            repeats = []
        result = None
        if repeats:
            result = {
                phase: {
                    metric: (
                        None
                        if repeats[0][phase][metric] is None
                        else float(np.median([r[phase][metric] for r in repeats]))
                    )
                    for metric in repeats[0][phase]
                }
                for phase in ("fit", "predict")
            }
            dev = result["predict"]["device_mb"]
            print(
                f"fit {result['fit']['time_s']:7.3f}s  predict {result['predict']['time_s']:7.3f}s  "
                f"RAM {result['predict']['ram_mb']:7.0f}MB" + ("" if dev is None else f"  device {dev:7.0f}MB")
            )
        config.update(device=str(device), model_path=None if self.model_path is None else str(self.model_path))
        self.records[key] = {"config": config, "result": result, "repeats": repeats}
        return result

    def run_sweeps(self, sweeps, device, offload_mode):
        """Run the train-rows / features / test-rows sweeps with and without kv cache.

        Returns ``{sweep_name: {cache_key: [result, ...]}}``.
        """
        results = {}
        for name, sweep in sweeps.items():
            results[name] = {}
            for cache_key, kv_cache in (("no_cache", False), ("cache", True)):
                rows = []
                for value in sweep["values"]:
                    sizes = {k: v for k, v in sweep.items() if k != "values"}
                    sizes[name] = value
                    rows.append(self.run(kv_cache=kv_cache, device=device, offload_mode=offload_mode, **sizes))
                results[name][cache_key] = rows
        return results


def _metric(results, phase, metric):
    """Extract a metric from a list of results (``None`` for missing points)."""
    out = []
    phases = ("fit", "predict") if phase == "fit+predict" else (phase,)
    for r in results:
        values = None if r is None else [r[p][metric] for p in phases]
        out.append(None if values is None or None in values else sum(values))
    return out


# --- Plotting ----------------------------------------------------------------


def _style_axes(fig, ytitle, mobile, log_x=True):
    fig.update_xaxes(
        type="log" if log_x else None,
        tickfont=TICK_FONT,
        zeroline=False,
        gridcolor=GRID_COLOR,
        title_font=TITLE_FONT,
    )
    fig.update_yaxes(
        type="log",
        tickfont=TICK_FONT,
        zeroline=False,
        gridcolor=GRID_COLOR,
        title_font=TITLE_FONT,
        title_standoff=6,
    )
    # Linked y-axes across subplots.
    fig.update_layout(
        {name: dict(matches="y") for name in fig.layout.to_plotly_json() if name[:5] == "yaxis" and name != "yaxis"}
    )
    # Desktop: y-axis title on the first subplot only; mobile: on every row.
    fig.update_yaxes(title_text=ytitle, row=1, col=1)
    if mobile:
        fig.update_yaxes(title_text=ytitle)


def _finalize_layout(
    fig, *, title, subtitle, footnote, height, margin_t, margin_b, legend_offset_px, legend_title=None
):
    """Title, subtitle, footnote and legend placement shared by all figures."""
    plot_height = height - margin_t - margin_b
    # Subplot titles a bit above the plotting area.
    for annotation in fig.layout.annotations:
        if annotation.yref == "paper":
            annotation.update(y=annotation.y + 0.03, font=dict(size=12, color="#1f2937"))
    fig.add_annotation(
        text=f"<b>{title}</b><br><span style='{SUBTITLE_STYLE}'>{subtitle}</span>",
        x=0.5,
        xanchor="center",
        xref="paper",
        y=1 + margin_t / plot_height,
        yanchor="top",
        yref="paper",
        showarrow=False,
        font=dict(size=18),
    )
    fig.add_annotation(
        text=f"<span style='{FOOTNOTE_STYLE}'>{footnote}</span>",
        x=1,
        xanchor="right",
        xref="paper",
        y=-(margin_b - 8) / plot_height,
        yanchor="bottom",
        yref="paper",
        showarrow=False,
    )
    legend = dict(
        font=dict(size=12),
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-legend_offset_px / plot_height,
        yanchor="top",
    )
    if legend_title:
        legend["title"] = dict(text=f"<b>{legend_title}</b>", font=dict(size=13))
    fig.update_layout(
        legend=legend,
        margin=dict(l=20, r=5, t=margin_t, b=margin_b),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        height=height,
    )
    return fig


def make_subplots_grid(n_panels, subplot_titles, mobile):
    from plotly.subplots import make_subplots

    if mobile:
        spacing = 0.18 if n_panels == 3 else 0.36
        return make_subplots(rows=n_panels, cols=1, subplot_titles=subplot_titles, vertical_spacing=spacing)
    spacing = 0.04 if n_panels == 3 else 0.08
    return make_subplots(rows=1, cols=n_panels, subplot_titles=subplot_titles, horizontal_spacing=spacing)


def _panel_position(i, mobile):
    return dict(row=i + 1, col=1) if mobile else dict(row=1, col=i + 1)


SWEEP_AXIS_TITLES = {"n_train": "Number of train rows", "n_features": "Number of features", "n_test": "Test-set size"}
SWEEP_PANEL_TITLES = {"n_train": "number of train rows", "n_features": "number of features", "n_test": "test-set size"}


def make_sweep_figure(
    sweeps, results, traces, *, quantity, ytitle, title, subtitle, footnote, hover_unit, hover_fmt, mobile
):
    """Three log-log panels (train rows / features / test rows), one line per trace.

    ``traces`` is a list of ``(cache_key, phase, metric, label, dash, width)``.
    """
    import plotly.graph_objects as go

    subplot_titles = []
    for name, sweep in sweeps.items():
        fixed = ", ".join(f"{k}={v}" for k, v in sweep.items() if k != "values")
        subplot_titles.append(f"{quantity} vs {SWEEP_PANEL_TITLES[name]}<br>({fixed})")
    fig = make_subplots_grid(3, subplot_titles, mobile)
    for i, (name, sweep) in enumerate(sweeps.items()):
        for cache_key, phase, metric, label, dash, width in traces:
            name_full = f"{CACHE_LABELS[cache_key]}, {phase.replace('+', ' + ')}"
            fig.add_trace(
                go.Scatter(
                    x=sweep["values"],
                    y=_metric(results[name][cache_key], phase, metric),
                    name=name_full,
                    legendgroup=name_full,
                    showlegend=i == 0,
                    mode="lines+markers",
                    line=dict(color=COLORS[cache_key], dash=dash, width=width),
                    marker=dict(color=COLORS[cache_key], size=7, symbol=SYMBOLS[cache_key]),
                    hovertemplate=(
                        f"{SWEEP_AXIS_TITLES[name]}=%{{x}}<br>{label}=%{{y:{hover_fmt}}} {hover_unit}"
                        f"<extra>{CACHE_LABELS[cache_key]}</extra>"
                    ),
                ),
                **_panel_position(i, mobile),
            )
        fig.update_xaxes(title_text=SWEEP_AXIS_TITLES[name], **_panel_position(i, mobile))
    _style_axes(fig, ytitle, mobile)
    height, margin_t = (808, 118) if mobile else (448, 118)
    return _finalize_layout(
        fig,
        title=title,
        subtitle=subtitle,
        footnote=footnote,
        height=height,
        margin_t=margin_t,
        margin_b=90,
        legend_offset_px=45,
    )


def save_figure(figures, number, output_dir, show=False):
    """Write the docs embed snippet (desktop + mobile) and a standalone preview."""
    desktop, mobile = figures
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for css_class, fig, include_plotlyjs in (("plotly-desktop", desktop, "cdn"), ("plotly-mobile", mobile, False)):
        inner = fig.to_html(
            full_html=False, include_plotlyjs=include_plotlyjs, default_height="100%", config={"responsive": True}
        )
        parts.append(
            f'<div class="{css_class}">\n<div style="height:{fig.layout.height}px; width:100%;">{inner}</div>\n</div>'
        )
    snippet = output_dir / f"fig{number}.html"
    snippet.write_text("\n".join(parts) + "\n")
    preview = output_dir / f"fig{number}_preview.html"
    desktop.write_html(preview, include_plotlyjs="cdn", config={"responsive": True})
    print(f"Figure {number} written to {snippet} (preview: {preview})")
    if show:
        desktop.show()


def save_results(benchmark, metadata, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    payload = {
        "metadata": metadata,
        "n_repeats": benchmark.n_repeats,
        "records": [
            {**r["config"], "result": r["result"], "repeats": r["repeats"]} for r in benchmark.records.values()
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Raw results written to {path}")


if RUN_BENCHMARKS:
    import torch

    from tabicl._torch_devices import resolve_torch_device

    ARGS = parse_args()
    if ARGS.worker is not None:
        run_worker(ARGS.worker)
        sys.exit(0)
    DEVICE = resolve_torch_device(ARGS.device)
    SWEEPS = {
        "n_train": QUICK_TRAIN_ROWS_SWEEP if ARGS.quick else TRAIN_ROWS_SWEEP,
        "n_features": QUICK_FEATURES_SWEEP if ARGS.quick else FEATURES_SWEEP,
        "n_test": QUICK_TEST_ROWS_SWEEP if ARGS.quick else TEST_ROWS_SWEEP,
    }
    BENCHMARK = Benchmark(n_repeats=ARGS.n_repeats, model_path=ARGS.model_path, isolate=not ARGS.no_isolate)
    METADATA = run_metadata(DEVICE)
    METADATA["model_path"] = None if ARGS.model_path is None else str(ARGS.model_path)
    METADATA["isolated_runs"] = BENCHMARK.isolate
    FOOTNOTE = footnote_text(METADATA)
    print(json.dumps(METADATA, indent=2))

# -----------------------------------------------------------------------------
# Figure 1: kv_cache effect on fit / predict time
# (section "Effects of ``kv_cache`` on device compute time")
# -----------------------------------------------------------------------------
if RUN_BENCHMARKS and 1 in ARGS.figures:
    SWEEP_RESULTS = BENCHMARK.run_sweeps(SWEEPS, DEVICE, ARGS.offload_mode)
    fig1_traces = [
        # (cache_key, phase, metric, hover label, dash, width)
        ("no_cache", "predict", "time_s", "predict", "dash", 2.0),
        ("no_cache", "fit+predict", "time_s", "fit + predict", "solid", 2.5),
        ("cache", "predict", "time_s", "predict", "dash", 2.0),
        ("cache", "fit+predict", "time_s", "fit + predict", "solid", 2.5),
    ]
    save_figure(
        [
            make_sweep_figure(
                SWEEPS,
                SWEEP_RESULTS,
                fig1_traces,
                quantity="Time",
                ytitle="Time (s)",
                title="Figure 1 — KV cache: predict and fit+predict time vs dataset size",
                subtitle=f"{device_label(DEVICE)}, {N_CLASSES} classes, n_estimators={N_ESTIMATORS}.",
                footnote=FOOTNOTE,
                hover_unit="s",
                hover_fmt=".3g",
                mobile=mobile,
            )
            for mobile in (False, True)
        ],
        1,
        ARGS.output_dir,
        show=ARGS.show,
    )
    save_results(BENCHMARK, METADATA, ARGS.output_dir)

# -----------------------------------------------------------------------------
# Figure 2: kv_cache effect on peak memory
# (section "Effects of ``kv_cache`` on memory")
# -----------------------------------------------------------------------------
if RUN_BENCHMARKS and 2 in ARGS.figures:
    # Reuses the measurements of figure 1 when available (memoized).
    SWEEP_RESULTS = BENCHMARK.run_sweeps(SWEEPS, DEVICE, ARGS.offload_mode)
    # On CPU there is no device memory: plot the peak host memory instead.
    on_cpu = DEVICE.type == "cpu"
    mem_metric, mem_name = ("ram_mb", "RAM") if on_cpu else ("device_mb", "VRAM")
    fig2_traces = [
        # (cache_key, phase, metric, hover label, dash, width)
        ("no_cache", "predict", mem_metric, f"{mem_name} predict", "dash", 2.5),
        ("cache", "predict", mem_metric, f"{mem_name} predict", "dash", 2.5),
        ("cache", "fit", mem_metric, f"{mem_name} fit", "solid", 3.5),
    ]
    save_figure(
        [
            make_sweep_figure(
                SWEEPS,
                SWEEP_RESULTS,
                fig2_traces,
                quantity=mem_name,
                ytitle=f"{mem_name} peak (MB)",
                title=(
                    "Figure 2 — KV cache: peak "
                    + ("host" if on_cpu else "device")
                    + " memory (fit, predict) vs dataset size"
                ),
                subtitle=f"{device_label(DEVICE)}, {N_CLASSES} classes, n_estimators={N_ESTIMATORS}.",
                footnote=FOOTNOTE,
                hover_unit="MB",
                hover_fmt=".0f",
                mobile=mobile,
            )
            for mobile in (False, True)
        ],
        2,
        ARGS.output_dir,
        show=ARGS.show,
    )
    save_results(BENCHMARK, METADATA, ARGS.output_dir)

# -----------------------------------------------------------------------------
# Figure 3: offload modes (``docs/_static/compute_tradeoffs/fig3.html``, not
# embedded in the gallery text at the moment)
# -----------------------------------------------------------------------------
if RUN_BENCHMARKS and 3 in ARGS.figures:
    import plotly.graph_objects as go

    offload_config = QUICK_OFFLOAD_CONFIG if ARGS.quick else OFFLOAD_CONFIG
    n_train, n_features = offload_config["n_train"], offload_config["n_features"]
    n_test_small, n_test_large = offload_config["n_test_small"], offload_config["n_test_large"]
    cpu = torch.device("cpu")
    accelerator = None if DEVICE.type == "cpu" else DEVICE
    acc_name = device_short_label(accelerator) if accelerator is not None else "GPU"
    offload_modes = ["gpu", "cpu", "disk"]

    def offload_bars(n_test, include_cpu):
        """Return ``(labels, results)`` for the bars of one panel."""
        labels, results = [], []
        if accelerator is not None:
            for mode in offload_modes:
                labels.append(f"{acc_name} — offload={mode}")
                results.append(BENCHMARK.run(n_train, n_features, n_test, False, accelerator, mode))
        if include_cpu:
            # offload_mode is a no-op on CPU.
            labels.append("CPU")
            results.append(BENCHMARK.run(n_train, n_features, n_test, False, cpu, "auto"))
        return labels, results

    panels = [
        offload_bars(n_test_small, include_cpu=accelerator is None or not ARGS.skip_cpu_baseline),
        offload_bars(n_test_large, include_cpu=False) if accelerator is not None else ([], []),
    ]
    # Device memory is called VRAM on GPUs; plotted as 0/missing for device='cpu'.
    dev_mem = "VRAM" if acc_name == "GPU" else f"{acc_name} memory"
    metrics = [
        # (legend name, legendgroup, metric, color, hover label)
        ("Predict time", "time", "time_s", COLORS["time"], "predict=%{y:.3g} s"),
        ("RAM peak", "ram", "ram_mb", COLORS["ram"], "RAM=%{y:.0f} MB"),
        (f"{dev_mem} peak", "vram", "device_mb", COLORS["vram"], f"{dev_mem}=%{{y:.0f}} MB"),
    ]

    def fmt_int(n):
        return f"{n:,}".replace(",", " ")

    # Device memory saved by offloading with the large test set, quoted in the subtitle.
    vram_large = {label: None if r is None else r["predict"]["device_mb"] for label, r in zip(*panels[1])}
    vram_gpu = vram_large.get(f"{acc_name} — offload=gpu")
    vram_offloaded = [v for k, v in vram_large.items() if v is not None and not k.endswith("offload=gpu")]
    if vram_gpu is not None and vram_offloaded:
        savings = f"offload modes save ~{(vram_gpu - min(vram_offloaded)) / 1024:.1f} GB {dev_mem}"
    else:
        savings = f"offload modes reduce {dev_mem}"
    subtitle = "<br>".join(
        [
            (
                f"Classification, {N_CLASSES} classes, n_estimators={N_ESTIMATORS}, kv_cache=False, "
                f"n_features={n_features}, n_train={n_train}."
            ),
            f"Left: n_test={fmt_int(n_test_small)}, {acc_name} vs CPU (offload no-op on CPU).",
            f"Right: n_test={fmt_int(n_test_large)}, {acc_name} only — large output tensors make {savings}.",
            f"Bars: predict time (black), RAM (blue), {dev_mem} (vermillion). Log y. "
            + (device_label(accelerator) if accelerator is not None else "No accelerator available."),
        ]
    )

    fig3 = []
    for mobile in (False, True):
        fig = make_subplots_grid(
            2,
            [
                f"n_test={fmt_int(n_test_small)}<br>({acc_name} vs CPU, n_train={n_train})",
                f"n_test={fmt_int(n_test_large)}<br>({acc_name} only, n_train={n_train})",
            ],
            mobile,
        )
        for i, (labels, results) in enumerate(panels):
            for name, group, metric, color, hover in metrics:
                y = [None if r is None else r["predict"][metric] for r in results]
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=y,
                        name=name,
                        legendgroup=group,
                        showlegend=i == 0,
                        marker=dict(color=color, line=dict(color="#1f2937", width=1)),
                        hovertemplate=f"%{{x}}<br>{hover}<extra></extra>",
                    ),
                    **_panel_position(i, mobile),
                )
            fig.update_xaxes(title_text="Device × offload mode", **_panel_position(i, mobile))
            if not labels:
                fig.add_annotation(
                    text="No accelerator available",
                    showarrow=False,
                    xref=f"x{i + 1 if i else ''} domain",
                    yref=f"y{i + 1 if i else ''} domain",
                    x=0.5,
                    y=0.5,
                )
        _style_axes(fig, "Predict time (s) / Memory (MB)", mobile, log_x=False)
        height, margin_t, margin_b, legend_offset = (740, 190, 150, 68) if mobile else (580, 190, 150, 105)
        fig3.append(
            _finalize_layout(
                fig,
                title="Figure 3 — Offload comparison: GPU vs CPU (left) and large n_test (right)",
                subtitle=subtitle,
                footnote=FOOTNOTE,
                height=height,
                margin_t=margin_t,
                margin_b=margin_b,
                legend_offset_px=legend_offset,
                legend_title="Metric",
            )
        )
    save_figure(fig3, 3, ARGS.output_dir, show=ARGS.show)
    save_results(BENCHMARK, METADATA, ARGS.output_dir)
# sphinx_gallery_end_ignore
