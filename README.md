# somsd (modernized)

A Self-Organizing Map / Graph-SOM (SOM-SD, PM-GraphSOM) trainer, updated
from CUDA 8 to CUDA 11.x, with a parallel (OpenMP) CPU-only build option
and a Python script to plot resulting winner-coordinate/label maps.

Original implementation: Markus Hagenbuchner et al. This is a modernization
pass on top of that codebase — see "What changed" below.

## Building

Requires `gcc`/`g++`, and (optionally) the CUDA 11.x toolkit for GPU support.

```
make               # auto-detect: builds with GPU support if nvcc is found,
                   # otherwise falls back to a CPU-only OpenMP build
make CPU_ONLY=1    # force a CPU-only OpenMP build even if nvcc is present
make clean
```

GPU builds target compute capability 5.0 through 8.6 (Maxwell through
Ampere) by default — edit `NVARCH` in the `Makefile` to narrow/widen that
if you know your target GPUs.

## Running

Same CLI as before (`-din`, `-cin`, `-cout`, `-xdim`, `-ydim`, `-iter`,
`-radius`, `-alpha`, `-res`, `-gpu[:N]`, etc — run `./somsd -help`).

New/changed for the CPU path:

- `-cpu <N>` sets the number of OpenMP threads used during CPU training
  (falls back to the `OMP_NUM_THREADS` environment variable, then to all
  cores, if not given).
- The two per-iteration hot loops — finding the best-matching codebook
  (`FindWinnerEucledian_CPU`) and updating codebook vectors
  (`GaussianAdapt_CPU`) — are now parallelized across codebooks. These
  loops scale with the number of codebook vectors (`xdim * ydim`), which is
  the dominant cost on high-resolution maps (e.g. a 3840x2160 grid is ~8.3M
  codebooks), so this is where the wall-clock time actually goes.

## Vector plot of results (`plot_som.py`)

`somsd -res results.txt ...` writes one line per training sample:
`<graph_id> <node_id> <winner_x> <winner_y> <label>`. `plot_som.py` turns
that into a plot of where samples landed on the map, colored by label:

```
pip install numpy matplotlib
python3 plot_som.py results.txt --out som_plot.png
python3 plot_som.py results.txt --xdim 3840 --ydim 2160 --out som_plot.png
```

- Small/medium maps: a true scatter plot with a label legend.
- High-resolution maps (default: >200,000 samples, override with
  `--scatter-limit`): automatically switches to a log-scaled density
  heatmap binned to the map grid, since plotting millions of individual
  markers isn't legible (or fast). Force either mode with `--mode scatter`
  or `--mode heatmap`.

Run with `-h` for all options (point size/alpha, DPI, custom title, etc).

## What changed from the CUDA 8 original

- **Build**: `Makefile` rewritten for CUDA 11.x (`-gencode` arch list
  covering sm_50 through sm_86, replacing the single hardcoded `-arch=sm_20`
  comment left over from Fermi-era CUDA). Added an explicit `CPU_ONLY=1`
  override so a CPU-only OpenMP build can be produced deliberately, not just
  as an automatic fallback when `nvcc` is missing.
- **NVML**: dropped the ~190KB vendored copy of `nvml.h` that used to ship
  in this repo; the Makefile now points at the CUDA toolkit's own
  `nvml.h`/`libnvidia-ml.so` instead of a stale private copy.
- **CPU path** (`train_cpu.c`): added OpenMP parallelism to the
  best-matching-codebook search and the codebook update step — see above.
  Wired `-cpu <N>` through to `omp_set_num_threads`.
- **GPU path** (`train_gpu.cu`, `cuda_utils.cu`): the CUDA kernels
  themselves used no deprecated/removed CUDA 8-only APIs (no texture
  references, no old Thrust calls), so they compile as-is under CUDA 11
  once modern `-gencode` flags are supplied. Fixed a couple of `%d`
  format-string bugs on `size_t` device-property fields
  (`totalGlobalMem`, `sharedMemPerBlock`, `memPitch`, `totalConstMem`) that
  would misprint on modern GPUs with >2GB memory, added missing `<cmath>`
  include, and extended the SM-version-to-core-count table
  (`_ConvertSMVer2Cores`) through Pascal/Volta/Turing/Ampere so
  `-gpu` auto-select reports sane core counts on current hardware.
- **New**: `plot_som.py` for visualizing training results (see above).

## Known limitations / not yet done

- Not compiled/tested against an actual CUDA 11 toolchain in this pass —
  review the `Makefile` and kernel code against your specific driver/nvcc
  version before relying on it, particularly the `-gencode` list (drop
  architectures you don't need to speed up compilation).
- `train.c` (the older/alternate training path referenced by `train.h`)
  was not modified.
- No automated tests exist for this codebase; validate against a known
  dataset before trusting results from a fresh build.
