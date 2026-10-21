# SWRC — spin-wave reservoir computer

A micromagnetic (mumax³) simulator for a **spin-wave-disk reservoir computer**:
a 1 µm circular magnetic disk driven by RF pulses through actuator regions, used
as the nonlinear feature map of a physical reservoir computer. This repository
contains the simulator, a minimal worked example that runs end-to-end on a
single GPU, a tutorial walkthrough, the input-data generators for the
water-wave reservoir tasks, and the analyses that back
**"Autonomous robotic operation controlled by wave-based neuromorphic hardware"**
(*Nature Communications*; Jacob Zohar, Daniele Pinna, Gerrit van der Laan,
Thorsten Hesjedal, C. K. Safeer).

## Layout

```
SWRC/
├── README.md                ← this file
├── LICENSE                  ← MIT license
├── PSEUDOCODE.md            ← formal algorithm specification (the four pipeline stages)
├── requirements.txt         ← pinned Python dependencies
├── simulator/               ← THE simulator — mumax³ code + sweep driver + cube assembler
│   ├── README.md
│   ├── GEOMETRY.md          ← canonical parameter table (incl. VCMA convention)
│   ├── run_sweep.py
│   ├── build_relax_cache.mx3
│   ├── test_sample.mx3
│   └── build_cube.py
├── examples/                ← runnable end-to-end examples
│   └── minimal_classification/                  ← 2x2 XOR, 4 samples, ~5 min on one GPU
├── tutorial/                ← single-notebook walkthrough of the simulator (~200 lines)
│   ├── README.md
│   └── SWRC_MuMax3.ipynb
└── Demo/                    ← runnable demonstrations of the software (one per manuscript figure)
    ├── README.md
    ├── 2_3_5_Robotic_output_classification_task/      ← full workflow: dataset → simulator → SVM (Fig. 2e)
    └── reservoir_metrics/                              ← intrinsic-substrate characterisations
        ├── README.md
        ├── xor_chequerboard/                              ← SWR + MLP + CNN on the XOR-chequerboard task (Supplementary Fig. 7)
        └── kernel_rank_cg/                                ← kernel rank / d95 / participation ratio vs spatial coarse-graining (Section S9, Fig. S9.2 + Table S9.1)
```

## System requirements

### Software dependencies and operating systems (including version numbers)

| Component | Version | Notes |
|-----------|---------|-------|
| Python | ≥3.10 (tested on 3.13.13) | |
| MuMax3 | v3.10 (or compatible) | <https://mumax.github.io/>. Required for any simulator step; not needed to regenerate the figures from the bundled `.npz` caches. Install instructions below. |
| numpy | 2.2.6 | Figure stage also verified under numpy 2.4.2 |
| opencv-python | 4.12.0 | Used by the kernel-rank analysis |
| scikit-learn | 1.8.0 | LinearSVC + cross-validation |
| torch | 2.11.0 | CPU build sufficient; published MLP/CNN runs were CPU |
| matplotlib | 3.10.8 | Figure rendering |
| pandas | (any recent) | Faster CSV reads in `build_cube.py` |

Operating systems used to produce the published results: **Ubuntu 22.04 LTS**
(simulator + analysis compute stages) and **Windows 11** (figure regeneration
and supplementary-table `.docx` builder). All pinned dependency versions are
listed in [`requirements.txt`](requirements.txt).

### Versions the software has been tested on

- Ubuntu 22.04 LTS, Python 3.13.13, with the `requirements.txt` versions above.
- Windows 11, Python 3.13.13, with the same `requirements.txt` versions; the
  figure-regeneration stage is also verified under numpy 2.4.2.

### Required non-standard hardware

- One CUDA-capable GPU (≥ RTX 20-series equivalent recommended) is required to
  run the simulator stage. The figure / analysis stages run on CPU; the demo
  also includes a no-GPU path (see *Demo* below) that exercises the install
  end-to-end without a GPU.
- No other non-standard hardware is required.

## Installation guide

### Instructions

#### 1. Python environment

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is grouped by experiment and by pipeline stage; the
compute stages were run on Linux, the figure stages also run on Windows.

#### 2. MuMax3 binary

MuMax3 is required only to run the simulator (Algorithm 1 in
[`PSEUDOCODE.md`](PSEUDOCODE.md)). To install:

1. Download a v3.10 build for your platform from <https://mumax.github.io/>
   (Linux: `mumax3.10_linux_cuda12.0.tar.gz`; Windows:
   `mumax3.10_windows_cuda12.0.zip`).
2. Extract the archive and either:
   - place the `mumax3` (Linux) / `mumax3.exe` (Windows) binary on your
     `PATH`, **or**
   - set the `MUMAX3_PATH` environment variable to the absolute path of the
     binary.
3. Verify with `mumax3 -test` (Linux) or `mumax3.exe -test` (Windows); the
   binary will print its build info and the CUDA device it sees.

### Typical install time on a "normal" desktop computer

**~7 minutes total**, broken down as:

- ~5 min for the Python environment (`pip install -r requirements.txt` on a
  normal broadband connection).
- ~2 min for the mumax3 binary download.

This assumes a working CUDA driver is already present. The CUDA driver
installation itself (if not present) is platform-specific and not counted
above; follow NVIDIA's instructions for your platform.

## Demo

### Instructions to run on data

Two demo paths are bundled. Run **either or both** — they exercise different
parts of the pipeline.

**GPU demo** — full end-to-end (substrate → cube → classifier) on a 4-sample
2 × 2 XOR task:

```bash
cd examples/minimal_classification
bash run_example.sh
```

**No-GPU demo** — regenerate Supplementary Fig. 7 from the bundled `.npz`
result caches; runs entirely on CPU and does not invoke mumax3:

```bash
cd Demo/reservoir_metrics/xor_chequerboard
python make_xor_comparison_figure.py
```

### Expected output

**GPU demo** writes the following into `examples/minimal_classification/`:

- `sweep_out/` — 4 mumax3 run directories + the relax cache (~20 MB total).
- `mini_cube.npy` — `(4, 101, 64, 64)` float32 `m_z` cube (~10 MB).
- `result.json` — leave-one-out balanced accuracy and per-fold record. On
  this 4-sample task the balanced accuracy is typically **1.000**
  (4 / 4 folds correct).
- `response_snapshot.png` — 4-panel `m_z` snapshot at the peak-MI frame.

The terminal also prints a one-line summary, e.g.:

```
[result] LOO-CV balanced accuracy = 1.000  (4/4 folds correct)
         top-K MI features used: K = 4  (peak MI = 0.918 bits / 1.000 bit ceiling)
         saved result.json + response_snapshot.png
```

**No-GPU demo** writes a single figure file:

- `Demo/reservoir_metrics/xor_chequerboard/figures/figS7_xor_ladder_comparison.png` —
  the Supplementary Fig. 7 panel (balanced accuracy vs FLOPs for the
  spin-wave reservoir, MLP-on-coords, and CNN-on-coords on the
  XOR-chequerboard task ladder).

### Expected run time for demo on a "normal" desktop computer

- **GPU demo:** ~5 minutes on a single RTX 3070 / RTX 5070-class GPU (relax
  cache ~30 s, 4-sample sweep ~2 min, cube build ~10 s, classifier +
  figure ~30 s). Older GPUs scale roughly with mumax3 throughput.
- **No-GPU demo:** ~5 seconds on a normal laptop CPU.

## Instructions for use

### How to run the software on your data

The pipeline has two stages: the simulator that produces an `m_z` cube from
a coordinate-encoded input grid, and the analysis stage that reads the cube
and produces a classification or dimensionality result.

**1. Generate an `m_z` cube from your own input grid.** The simulator stage
is configured entirely via environment variables (see the file header of
[`simulator/run_sweep.py`](simulator/run_sweep.py) for the full list and
defaults):

```bash
cd simulator
SWRC_GRID=16 SWRC_TMAX=4e-9 SWRC_N_GPUS=1 python run_sweep.py
python build_cube.py --src sweep_out --out my_cube.npy --frames 201
```

Output is a `(N_samples, T, H, W)` float32 `.npy` cube plus a
`run_index.json` that maps cube rows to `(X1, X2)` input coordinates. The
full physical model and parameter reference is in
[`simulator/GEOMETRY.md`](simulator/GEOMETRY.md); a minimal walkthrough of
the substrate physics in inline Python is in
[`tutorial/SWRC_MuMax3.ipynb`](tutorial/SWRC_MuMax3.ipynb).

**2. Run a demonstration on your cube.** Each demo in `Demo/` exposes a
`--cube` CLI argument or `CUBE_PATH` environment variable to redirect the
analysis at your own cube. The three bundled demos:

- [`Demo/2_3_5_Robotic_output_classification_task/`](Demo/2_3_5_Robotic_output_classification_task/) —
  full obstacle-classification workflow: dataset generation (three
  notebooks for the 2-, 3- and 5-class variants), instructions for
  feeding the dataset to the simulator, expected simulator output, and
  a linear-SVM training + testing script
  (`sklearn.svm.SVC(kernel='linear')`) that produces the Fig. 2e
  comparison panels (`FinalResultsMain`, `SWvsSVMW`, `WWvsSVMvsSW`).
- [`Demo/reservoir_metrics/xor_chequerboard/`](Demo/reservoir_metrics/xor_chequerboard/) —
  SWR + MLP + CNN on an XOR-chequerboard task ladder. Generates
  Supplementary Fig. 7.
- [`Demo/reservoir_metrics/kernel_rank_cg/`](Demo/reservoir_metrics/kernel_rank_cg/) —
  kernel rank, d95, and participation ratio of the SWR feature matrix
  vs spatial coarse-graining. Generates Supplementary Table 2 / Table
  S9.1 and Fig. S9.2.

See each demo's `README.md` for the exact CLI / env-var knobs, the
bundled inputs, and the data shape it expects.

### Reproduction instructions

Every figure in the manuscript regenerates from the bundled `.npz` result
caches without the multi-GB `m_z` cubes; they run in seconds on a laptop.

| Manuscript figure | Command (from repo root) | Output |
|-------------------|--------------------------|--------|
| **Fig. 2e** (WW vs Raw SVM vs SWRC training-size sweep) | `cd Demo/2_3_5_Robotic_output_classification_task && python train_test_svm.py` | `figures/FinalResultsMain.png`, `figures/SWvsSVMW.png`, `figures/WWvsSVMvsSW.png` |
| **Supplementary Fig. 7** (XOR ladder, SWR vs MLP vs CNN) | `cd Demo/reservoir_metrics/xor_chequerboard && python make_xor_comparison_figure.py` | `figures/figS7_xor_ladder_comparison.png` |
| **Section S9.1** (kernel rank, d95, PR vs spatial CG — display panel) | `cd Demo/reservoir_metrics/kernel_rank_cg && python make_fig_s91.py` | `figures/figS91_kr_pr.png` |
| **Fig. S9.2** (per-frame spatial rank) | `cd Demo/reservoir_metrics/kernel_rank_cg && python make_fig_spatial.py` | `figures/figS92_spatial_rank.png` |
| Supporting analysis figs 1–5 (rank, SV spectra, cum. variance, multi-metric rank, CG montage) | `cd Demo/reservoir_metrics/kernel_rank_cg && python make_figures.py` | `figures/fig{1..5}_*.png` |
| **Table S9.1** / **Supplementary Table 2** (.docx, Windows + MS Office only) | `cd Demo/reservoir_metrics/kernel_rank_cg && python build_docx_s91.py` | `SI_S9_1_Kernel_Rank.docx` |

End-to-end regeneration *including* the simulator step (for one demo at a
time) follows the recipe in each demo's `README.md`. The simulator stage
requires mumax3 + a GPU and takes hours on multiple GPUs for the
production sweeps; see [`simulator/README.md`](simulator/README.md).

## Data availability

The large spin-wave (`m_z`) micromagnetic data cubes the experiments consume
are **not** bundled here (each is multiple GB). They are reproducible from the
simulator — every cube was generated by [`simulator/run_sweep.py`](simulator/run_sweep.py)
+ [`simulator/build_cube.py`](simulator/build_cube.py) with the parameters
documented in [`simulator/GEOMETRY.md`](simulator/GEOMETRY.md). The
precomputed `.npz` result caches **are** bundled in each experiment folder,
so every published figure regenerates without the raw cubes. The cubes
themselves are available from the corresponding author
(safeer.chenattukuzhiyil@physics.ox.ac.uk) on reasonable request.

## Code description / pseudocode

A formal specification of the four pipeline stages — the substrate
simulation sweep, the cube assembler, the top-`K` mutual-information +
LinearSVC readout, and the kernel-rank / d95 / participation-ratio
analysis — is in [`PSEUDOCODE.md`](PSEUDOCODE.md). Each algorithm block is
paired with a pointer to the canonical Python / mumax³ implementation in
this repository.

## License

This software is released under the MIT License — see [`LICENSE`](LICENSE) for
the full text. The MIT License is an
[Open Source Initiative–approved](https://opensource.org/licenses/MIT)
permissive licence: users may use, copy, modify, redistribute, and incorporate
the code in derivative works (including closed-source ones), subject only to
preservation of the copyright notice and the licence text.

---

---

## Component-by-component technical review and ideas — 2026-10-21

> **Scope:** This is a documentation-only review of the checked-in code, not a claim that any suggested change was implemented or new simulations were executed. No source code, notebooks, datasets or figures are modified in this branch. The **when** field means a stage in the execution workflow, not an empirically measured runtime.

### How the software flows

**Physics:** `simulator/build_relax_cache.mx3` → `simulator/run_sweep.py` → per-sample magnetic CSVs → `simulator/build_cube.py` → an indexed magnetization cube.

**Minimal test:** `examples/minimal_classification/run_example.sh` runs the preceding stages and then `examples/minimal_classification/train_minimal.py`.

**Robot classification:** three generator notebooks define distance tasks; pre-extracted SWRC features and a separate measured water-wave CSV feed `Demo/2_3_5_Robotic_output_classification_task/train_test_svm.py`.

**XOR comparison:** `make_dataset.py` creates coordinates and labels; `build_xor_pixel_mi.py` ranks substrate responses; `top_k_svm_sweep.py` evaluates readouts. `run_mlp_sweep_xor.py`, `run_cnn_sweep_xor.py` and `run_ext_sweep_xor.py` provide neural baselines; `analyse_xor.py` and `make_xor_comparison_figure.py` summarize them.

**Dimensionality:** `kernel_rank_cg.py` and `spatial_rank_per_frame.py` measure singular-value properties of large magnetization cubes; the figure and DOCX scripts render the smaller cached results.

**Scientific boundary:** outputs of MuMax3 are simulated magnetic behavior. Estimated software FLOPs are not physical device energy or latency. Rerendering a figure from bundled NPZ caches is not the same as rerunning the raw physical simulation or independently measuring hardware.

### Highest-priority observations

| Priority | Files | Why this matters | Future action (not implemented) |
|---|---|---|---|
| P0 | `examples/minimal_classification/train_minimal.py` | Supervised MI voxels are chosen before leave-one-out splits, using test labels. | Select features on training folds only and re-evaluate. |
| P0 | `build_xor_pixel_mi.py` and `top_k_svm_sweep.py` | Full-label MI mapping precedes CV and can inflate held-out performance. | Fold-local quantiles and MI ranking; keep global MI only for descriptive analysis. |
| P1 | `train_test_svm.py` | Three-class input contains 999 valid rows, while the loader zero-pads arrays to 1,000. | Preserve valid per-task lengths and verify SWRC feature/sample alignment. |
| P1 | `simulator/run_sweep.py` | Resume test uses at least N_FRAMES but regular output spans N_FRAMES+1 times. | Verify exact frames and per-run parameter signatures. |
| P1 | `simulator/run_sweep.py` | Current sweep is two-input; the robotic demo describes six driven sensor channels. | Implement and validate a separate six-input configuration in future code work. |
| P1 | `top_k_svm_sweep.py` and figure scripts | A 2K linear-readout operation estimate is not whole-system physical energy. | State training, readout, sensing and substrate costs separately. |
| P2 | `PSEUDOCODE.md` | Algorithm 4 points to an experiments/ path; rank implementation lives in Demo/reservoir_metrics/kernel_rank_cg/. | Reconcile the stale path in a future docs pass. |

### Every executable file: what / why / where / when / how / ideas

Each file path below links to a checked-in file in this repository. **Idea** is prospective; **validation** states how a later change could be tested.

#### Physics and simulator

**1. [`simulator/build_relax_cache.mx3`](simulator/build_relax_cache.mx3)**

- **What:** Builds the fixed magnetic geometry and equilibrium magnetization cache.
- **Why:** Every swept input should begin from the same physical state.
- **Where:** MuMax3 setup; produces an OVF equilibrium state.
- **When:** Once before a sweep, and again after physics parameters change.
- **How:** Defines the 512×512×1 mesh, material parameters, six actuator regions, bias field, relaxation and OVF save.
- **Idea and rationale:** Record the geometry, field, anisotropy, MuMax3 version and final energy beside a checksum; invalidate outdated relaxation caches.
- **Validation plan:** Compare independent relaxations with different initial states and verify convergence.

**2. [`simulator/test_sample.mx3`](simulator/test_sample.mx3)**

- **What:** Runs a manually specified two-input case (20,12).
- **Why:** Diagnoses pulses, solver timing and output shape without a grid sweep.
- **Where:** Manual physical-model smoke test.
- **When:** Before large-scale GPU computation or when troubleshooting a sample.
- **How:** Loads a precomputed relaxed OVF, applies time-delayed Gaussian-envelope 800 MHz anisotropy pulses and saves m_z every 20 ps for 6 ns.
- **Idea and rationale:** Use a shared physics configuration and explicitly document differences from sweep defaults, including coordinate step and simulation duration.
- **Validation plan:** Compare a matched sample against the programmatic driver; ensure the manually required relax_init.ovf is present.

**3. [`simulator/run_sweep.py`](simulator/run_sweep.py)**

- **What:** Runs all coordinate pairs and converts their simulated spin-wave responses into spatially reduced CSV frames.
- **Why:** This creates the nonlinear physical feature mapping used by downstream analyses.
- **Where:** Main GPU simulation; creates relaxation cache, per-sample directories, grid_int.npy and a manifest.
- **When:** For each coordinate pair after equilibrium preparation; can skip previously completed runs.
- **How:** Generates sample MuMax3 scripts, dispatches per-GPU worker processes, reads binary OVFs, crops magnetization and block-averages 384×384 to 64×64.
- **Idea and rationale:** P1: require exactly the expected output frame count and a run-parameter checksum. Current skip test accepts at least N_FRAMES although the nominal time axis contains N_FRAMES+1 snapshots. Expose six driven inputs for the robotic sensor task rather than assuming two.
- **Validation plan:** Interrupt and resume a toy grid, check zero-to-Tmax time alignment and confirm every run has the same output shape.

**4. [`simulator/build_cube.py`](simulator/build_cube.py)**

- **What:** Assembles simulation CSV frames into a float32 sample×time×height×width cube and run index.
- **Why:** Downstream MI and classifiers need an unambiguous sample-to-coordinate mapping.
- **Where:** Between simulator outputs and all analysis scripts; writes an NPY cube plus run_index JSON.
- **When:** After generating complete per-sample CSV folders.
- **How:** Sorts sample directories, parses coordinate names, validates frame sizes/counts, stacks data and reports dead-cell statistics.
- **Idea and rationale:** P1: cross-check coordinates against grid_int.npy and reject duplicate/missing samples; add streaming/memory-mapped output for large sweeps.
- **Validation plan:** Use a miniature fixture with deliberately shuffled directories, missing frames and non-square frames.

#### Minimal example and tutorial

**5. [`examples/minimal_classification/run_example.sh`](examples/minimal_classification/run_example.sh)**

- **What:** Chains the four-sample MuMax3 sweep, cube assembler and XOR classifier.
- **Why:** It is the smallest full-pipeline installation and reproducibility test.
- **Where:** Example orchestration; writes sweep_out, mini_cube.npy, result.json and response_snapshot.png.
- **When:** After installing Python dependencies and MuMax3.
- **How:** Sets the grid to 2, duration to 2 ns and one GPU, then calls three scripts with consistent expected dimensions.
- **Idea and rationale:** Add preflight checks for GPU availability, MuMax3 version, output space and incomplete files; preserve a full parameter manifest.
- **Validation plan:** Run end to end, verify the expected four samples and repeat without recomputing correct runs.

**6. [`examples/minimal_classification/train_minimal.py`](examples/minimal_classification/train_minimal.py)**

- **What:** Ranks response voxels by mutual information and classifies XOR labels with LinearSVC.
- **Why:** It demonstrates nonlinear reservoir features followed by a linear readout.
- **Where:** Minimal analysis/readout, JSON metrics and response visualization.
- **When:** After a valid four-sample cube is built.
- **How:** Estimates quantile-binned voxel MI, selects top-K time/space indices, standardizes features and performs leave-one-out classification.
- **Idea and rationale:** P0: supervised MI ranking currently occurs on all four labels before leave-one-out CV; move feature selection and bin fitting inside each training fold before interpreting test accuracy.
- **Validation plan:** Compare existing evaluation to train-fold-only selection; add shuffled-label controls and uncertainty warnings for N=4.

**7. [`tutorial/SWRC_MuMax3.ipynb`](tutorial/SWRC_MuMax3.ipynb)**

- **What:** Shows how to construct MuMax3 simulation inputs and read responses interactively.
- **Why:** Provides an educational, inspectable version of the simulation procedure.
- **Where:** Tutorial notebook alongside production simulator.
- **When:** While onboarding or exploring a single experiment.
- **How:** Builds solver-script text from Python parameters, launches the solver and inspects output arrays.
- **Idea and rationale:** Add a comparison cell proving whether tutorial geometry, pulse timing and field values match production; label differences clearly.
- **Validation plan:** Run identical single-sample settings across notebook and simulator and compare frame/time axes.

#### Robot distance-classification demonstration

**8. [`Demo/2_3_5_Robotic_output_classification_task/2_output_unique_data_generation.ipynb`](Demo/2_3_5_Robotic_output_classification_task/2_output_unique_data_generation.ipynb)**

- **What:** Generates distinct, class-balanced six-sensor vectors and two labels.
- **Why:** Establishes the simplest binary obstacle decision.
- **Where:** Data-generation stage; associated with Train2X.csv and Train2Y.csv.
- **When:** Before encoding sensor readings or training the classifier.
- **How:** Samples six integers from 1–6; the positive class has at least one entry in 1–3.
- **Idea and rationale:** Log dataset seed, admissible population size, uniqueness checks and split provenance.
- **Validation plan:** Assert six columns, allowed range, no duplicate rows and balanced labels.

**9. [`Demo/2_3_5_Robotic_output_classification_task/3_output_unique_data_generation.ipynb`](Demo/2_3_5_Robotic_output_classification_task/3_output_unique_data_generation.ipynb)**

- **What:** Creates left/right-sensitive three-class obstacle examples.
- **Why:** Tests localization beyond the binary detection task.
- **Where:** Data-generation stage; associated with Train3X.csv and Train3Y.csv.
- **When:** Before simulation or classification.
- **How:** Checks the first versus last three sensors for near-distance values, excluding conflicting patterns.
- **Idea and rationale:** Clearly quantify exclusions and enforce the actual 999-row valid dataset size downstream.
- **Validation plan:** Regenerate a deterministic set and verify 333 examples per class and label exclusivity.

**10. [`Demo/2_3_5_Robotic_output_classification_task/5_output_unique_data_generation.ipynb`](Demo/2_3_5_Robotic_output_classification_task/5_output_unique_data_generation.ipynb)**

- **What:** Builds clear, near-left, near-right, mid-left and mid-right tasks.
- **Why:** Tests increased label granularity from six sensors.
- **Where:** Data-generation stage; associated with Train5X.csv and Train5Y.csv.
- **When:** Before six-input encoding.
- **How:** Uses 1–7-valued distances, near bands 1–2, mid bands 3–4, and explicit admissibility filters.
- **Idea and rationale:** Publish the priority policy for simultaneous detections and report rejected combinations.
- **Validation plan:** Check all five class counts and hand-coded edge cases.

**11. [`Demo/2_3_5_Robotic_output_classification_task/train_test_svm.py`](Demo/2_3_5_Robotic_output_classification_task/train_test_svm.py)**

- **What:** Compares linear SVMs on six raw readings and cached 2,500-feature spin-wave representations; plots water-wave laboratory baseline.
- **Why:** It supplies the reported recognition-versus-training-size comparisons for 2, 3 and 5 output classes.
- **Where:** Figure 2e pipeline; reads paired Train CSVs, swrc_features.npz and training_frame3_results.csv.
- **When:** After inputs and pre-extracted wave features exist.
- **How:** Reshuffles samples, trains a linear-kernel SVC, tests on the following 100 rows, repeats and plots means/spread.
- **Idea and rationale:** P1: the three-class source has 999 rows but loader allocates 1,000 and leaves a zero-filled entry. Preserve per-task valid lengths and verify cached feature row alignment; clearly identify laboratory results as imported, not recomputed.
- **Validation plan:** Re-evaluate after removing padding, check same sample/label mapping in both feature arms and save exact train/test indices.

#### XOR chequerboard benchmark

**12. [`Demo/reservoir_metrics/xor_chequerboard/make_dataset.py`](Demo/reservoir_metrics/xor_chequerboard/make_dataset.py)**

- **What:** Produces 16×16 and 32×32 normalized coordinate arrays with parity-chequerboard labels at multiple periods.
- **Why:** Provides controlled nonlinear tasks with precisely known class balance.
- **Where:** Stage 0; writes data/xor_datasets.npz.
- **When:** Before MI or neural-network training.
- **How:** Computes parity from integer divisions by scale P and flattens it in grid order.
- **Idea and rationale:** Include explicit sample-order metadata, and consider spatial-shifted/held-out-grid variants when testing transfer rather than interpolation.
- **Validation plan:** Check balance, coordinate normalization and agreement with the cube row index.

**13. [`Demo/reservoir_metrics/xor_chequerboard/build_xor_pixel_mi.py`](Demo/reservoir_metrics/xor_chequerboard/build_xor_pixel_mi.py)**

- **What:** Computes voxel-by-voxel mutual information with XOR labels for each scale.
- **Why:** Selects physically informative spatiotemporal response sites.
- **Where:** Stage 1a; reads raw wave-response cube and stores MI volumes and summary NPZs.
- **When:** After MuMax3 cube production and before top-K feature extraction.
- **How:** Quantile-bins responses with five bins, counts bin×class cells, bias-corrects MI and records maxima.
- **Idea and rationale:** P0: distinguish full-dataset MI visualization from fold-local supervised ranking used for predictive evaluation. Avoid eager loading of multi-GB arrays when a chunked estimator is sufficient.
- **Validation plan:** Compare chunked and dense outputs on a small test cube; bootstrap MI and run label permutations.

**14. [`Demo/reservoir_metrics/xor_chequerboard/top_k_svm_sweep.py`](Demo/reservoir_metrics/xor_chequerboard/top_k_svm_sweep.py)**

- **What:** Measures LinearSVC accuracy for top-K selected voxels under analog/binary/ternary readouts.
- **Why:** Studies readout simplicity versus task performance.
- **Where:** Stage 1b; reads wave and MI arrays, writes topk_svm result NPZ.
- **When:** After per-scale MI computation.
- **How:** Selects highest-MI time/space voxels, extracts feature columns and evaluates standardized/quantized linear SVMs across repeated stratified folds.
- **Idea and rationale:** P0: MI ranking was computed with all labels before CV; select features separately within training folds. Account separately for feature-search effort and physical device cost rather than equating 2K inference FLOPs to system effort.
- **Validation plan:** Compare global and leakage-free nested feature selection on identical folds; test random-feature and permuted-label controls.

**15. [`Demo/reservoir_metrics/xor_chequerboard/run_sw_reservoir_sweep.sh`](Demo/reservoir_metrics/xor_chequerboard/run_sw_reservoir_sweep.sh)**

- **What:** Launches MI extraction followed by top-K SVM for five 16×16 drive settings.
- **Why:** Avoids manual orchestration mistakes across bias/frequency conditions.
- **Where:** Shell entry point to the SWRC arm of the XOR benchmark.
- **When:** When the associated multi-GB cubes are available in CUBE_DIR.
- **How:** Selects named cube files and invokes the two Python stages sequentially per drive.
- **Idea and rationale:** Validate all paths/configurations up front; log data hashes, per-drive elapsed time and restart markers.
- **Validation plan:** Ensure a missing or wrong-grid input fails immediately and does not yield a misleading partial summary.

**16. [`Demo/reservoir_metrics/xor_chequerboard/run_mlp_sweep_xor.py`](Demo/reservoir_metrics/xor_chequerboard/run_mlp_sweep_xor.py)**

- **What:** Trains a depth/width grid of coordinate-only MLPs on the XOR tasks.
- **Why:** Provides a conventional nonlinear software baseline.
- **Where:** Stage 2a; saves mlp_sweep_xor.npz.
- **When:** After the XOR dataset is prepared, independent of physical cube simulation.
- **How:** Uses repeated stratified CV, fixed-seed architecture training, Adam/cosine schedule, 400 epochs and per-architecture metrics.
- **Idea and rationale:** Add robust resume from the periodic checkpoint; compare other activations, positional encodings, seeds and matched tuning budgets.
- **Validation plan:** Re-run selected architectures and verify accuracy curves, parameter counts and operation estimates.

**17. [`Demo/reservoir_metrics/xor_chequerboard/run_cnn_sweep_xor.py`](Demo/reservoir_metrics/xor_chequerboard/run_cnn_sweep_xor.py)**

- **What:** Trains coordinate-image CNN architectures on the checkerboard task ladder.
- **Why:** Tests whether spatial inductive bias narrows the difference to wave computation.
- **Where:** Stage 2b; saves cnn_sweep_xor.npz.
- **When:** After coordinate/label data exists.
- **How:** Uses 3×3 padded convolutions on two-channel coordinate images under repeated stratified folds.
- **Idea and rationale:** For spatial fairness, compare random-pixel folds with spatial-block holdouts and document receptive-field access to neighboring pixels.
- **Validation plan:** Ensure no held-out labels enter gradients; record model-specific compute and deterministic results.

**18. [`Demo/reservoir_metrics/xor_chequerboard/run_ext_sweep_xor.py`](Demo/reservoir_metrics/xor_chequerboard/run_ext_sweep_xor.py)**

- **What:** Tests larger MLP/CNN architectures on fine-resolution 32×32 parity tasks.
- **Why:** Checks whether baseline shortcomings stem from limited model capacity.
- **Where:** Stage 2c; saves mlp_ext_xor.npz and cnn_ext_xor.npz.
- **When:** After initial architecture sweeps suggest difficult P=1,2,4 tasks.
- **How:** Extends depth/width combinations under similar optimizer and cross-validation settings.
- **Idea and rationale:** Include Fourier-feature or sinusoidal baselines and matched optimization budgets; do not interpret a finite search as proving an impossibility.
- **Validation plan:** Cross-check splits, scaling, labels and FLOP formulas with the original sweep scripts.

**19. [`Demo/reservoir_metrics/xor_chequerboard/analyse_xor.py`](Demo/reservoir_metrics/xor_chequerboard/analyse_xor.py)**

- **What:** Aggregates SWRC and neural baseline accuracy, model-size and FLOP caches into summary tables.
- **Why:** Transforms detailed runs into reproducible comparisons.
- **Where:** Stage 3; writes summary JSON/CSV and wavelength-match tables.
- **When:** After stage 1 and 2 caches are present.
- **How:** Loads NPZ results, averages folds, finds winning K/readouts and computes modeled training and inference work.
- **Idea and rationale:** P1: selecting a best-performing K or readout from the same evaluation folds can bias winning scores; introduce nested selection and preserve confidence intervals and provenance.
- **Validation plan:** Unit-test aggregation with synthetic arrays that have known winners and missing entries.

**20. [`Demo/reservoir_metrics/xor_chequerboard/make_xor_comparison_figure.py`](Demo/reservoir_metrics/xor_chequerboard/make_xor_comparison_figure.py)**

- **What:** Plots XOR-task accuracy versus modeled computation and reports smallest matching feed-forward models.
- **Why:** Visualizes the claimed accuracy/compute tradeoff.
- **Where:** Stage 4; writes figS7_xor_ladder_comparison.png.
- **When:** After cached sweep arrays have been aggregated.
- **How:** Loads SWRC, MLP, CNN and optional extended caches, forms Pareto frontiers and renders the five period panels.
- **Idea and rationale:** Label FLOPs explicitly as an estimate and separate reservoir sensing/drive costs from the linear readout; show statistical uncertainty and bounded search ranges.
- **Validation plan:** Verify every curve and threshold against source NPZ arrays in a clean reproduction run.

#### Reservoir rank and spatial resolution

**21. [`Demo/reservoir_metrics/kernel_rank_cg/kernel_rank_cg.py`](Demo/reservoir_metrics/kernel_rank_cg/kernel_rank_cg.py)**

- **What:** Estimates rank, explained-variance dimension and participation ratio versus spatial coarse-graining.
- **Why:** Characterizes how many distinct response modes survive reduced spatial sensing.
- **Where:** Rank compute stage; outputs kernel_rank_results.npz and montage_data.npz.
- **When:** After the full 256-sample spin-wave cube exists.
- **How:** Resizes frames using area averaging, centers full spatiotemporal feature vectors, computes economy SVD and tracks rank and variance metrics.
- **Idea and rationale:** Quantify noise-floor sensitivity and sample uncertainty; consider a smaller Gram-matrix eigenvalue method to reduce memory.
- **Validation plan:** Cross-check SVD values against centered sample Gram matrices and verify the native-resolution identity case.

**22. [`Demo/reservoir_metrics/kernel_rank_cg/spatial_rank_per_frame.py`](Demo/reservoir_metrics/kernel_rank_cg/spatial_rank_per_frame.py)**

- **What:** Measures effective spatial rank independently at every frame.
- **Why:** Separates spatial diversity from spatiotemporal diversity.
- **Where:** Per-frame rank compute; writes spatial_rank_results.npz.
- **When:** After the raw magnetization cube is available.
- **How:** Coarse-grains each frame, performs SVD on sample×spatial-feature matrices, stores rank curves and identifies maximum-d95 frames.
- **Idea and rationale:** Report median/prespecified frames beside the best-frame upper envelope; state the ceiling min(g², N−1).
- **Validation plan:** Compare best, typical and fixed frames; confirm rank cannot exceed the centered-data limit.

**23. [`Demo/reservoir_metrics/kernel_rank_cg/make_figures.py`](Demo/reservoir_metrics/kernel_rank_cg/make_figures.py)**

- **What:** Builds rank, singular-value, explained-variance, multi-metric and coarse-graining montage figures.
- **Why:** Makes cached dimensionality outputs visually auditable.
- **Where:** Figure-generation stage; writes multiple fig*.png files.
- **When:** After kernel_rank_results.npz and montage_data.npz exist.
- **How:** Uses Matplotlib to transform saved spectra, metrics and images into report plots.
- **Idea and rationale:** Annotate finite-sample limits, singular-value cutoff and selected montage frame; include uncertainty where available.
- **Validation plan:** Match every plotted array to its stored data and preserve figure provenance.

**24. [`Demo/reservoir_metrics/kernel_rank_cg/make_fig_s91.py`](Demo/reservoir_metrics/kernel_rank_cg/make_fig_s91.py)**

- **What:** Renders a supplemental summary of d95, participation ratio and cumulative variance.
- **Why:** Provides the compact S9.1 dimensionality comparison.
- **Where:** Cached-analysis figure stage; produces figS91_kr_pr.png.
- **When:** After kernel-rank results are saved.
- **How:** Reads grids, spectral cumulative variance and rank summaries and plots them.
- **Idea and rationale:** Add theoretical rank-ceiling guides and explain threshold-rank versus variance-rank semantics.
- **Validation plan:** Recompute d95 from the cumulative variance and check plotted grid order.

**25. [`Demo/reservoir_metrics/kernel_rank_cg/make_fig_spatial.py`](Demo/reservoir_metrics/kernel_rank_cg/make_fig_spatial.py)**

- **What:** Plots frame-resolved spatial dimensionality and best-frame versus full-volume rank.
- **Why:** Shows which temporal windows have high spatial complexity.
- **Where:** Figure stage; produces figS92_spatial_rank.png.
- **When:** After per-frame and full-volume rank NPZs exist.
- **How:** Loads both caches and renders time/grid comparisons.
- **Idea and rationale:** Label best-frame selection as a maximum over frames, not a typical operating point; add average and fixed-frame traces.
- **Validation plan:** Validate maxima, frame indices and theoretical spatial rank ceilings.

**26. [`Demo/reservoir_metrics/kernel_rank_cg/build_docx_s91.py`](Demo/reservoir_metrics/kernel_rank_cg/build_docx_s91.py)**

- **What:** Produces a Word supplement containing editable Office equations and rank figures.
- **Why:** Preserves equations, evidence and tables in a shareable scientific report.
- **Where:** Supplementary documentation/export stage.
- **When:** After numerical analysis and figure generation.
- **How:** Converts LaTeX via MathML/OMML, builds Word sections and embeds images/captions.
- **Idea and rationale:** Add cross-platform equation-conversion checks, source-cache provenance and automated numeric-table reconciliation.
- **Validation plan:** Open/export the DOCX on the target platform and compare all figures/numbers with source NPZs.

### Documentation, data and artifacts (not source code)

| Files or paths | Role, use and proposed improvement |
|---|---|
| `README.md`, `PSEUDOCODE.md` | Repository guide and formal algorithms; keep relative source links correct and clarify which results are simulated versus hardware-measured. |
| `simulator/README.md`, `simulator/GEOMETRY.md` | Operational instructions and physics reference; use one parameter manifest with explicit physical units and cache invalidation. |
| `examples/README.md`, `examples/minimal_classification/README.md`, `tutorial/README.md` | Onboarding references; identify reproducible commands and clearly state limits of four-sample tests. |
| `Demo/README.md`, `Demo/reservoir_metrics/README.md` and experiment-specific README files | Experiment recipes; cross-link each cached result to its actual source script, parameters and required raw cube. |
| `requirements.txt`, `.gitignore`, `LICENSE` | Dependency versions, ignored generated outputs and reuse rights; add platform tests and integrity conventions without modifying these files in this branch. |
| `Demo/2_3_5_Robotic_output_classification_task/data/Train{2,3,5}{X,Y}.csv` | Paired original distance vectors/labels. Track per-task row count and row order; three-class task has 999 valid examples. |
| `Demo/2_3_5_Robotic_output_classification_task/data/swrc_features.npz` | Cached 2,500-dimensional wave features; require an explicit six-input simulation and extraction recipe with sample IDs before claiming raw recomputation. |
| `Demo/2_3_5_Robotic_output_classification_task/data/training_frame3_results.csv` | Imported water-wave lab scores, not recomputed by the SVM script; retain measurement provenance and scoring protocol. |
| `Demo/reservoir_metrics/xor_chequerboard/data/xor_datasets.npz` | XOR coordinate/label sets; record flattening, normalization and dataset hash. |
| `Demo/reservoir_metrics/xor_chequerboard/data/pixel_mi_xor_summary_*.npz` and `topk_svm_*.npz` | MI and SWRC readout caches. Record MI selection protocol, original raw-cube hash, drive conditions and CV split IDs. |
| `Demo/reservoir_metrics/xor_chequerboard/data/{mlp,cnn}_sweep_xor.npz` and `{mlp,cnn}_ext_xor.npz` | MLP/CNN benchmark caches; archive architecture/optimizer seeds, FLOP assumptions and hardware timings. |
| `Demo/reservoir_metrics/xor_chequerboard/data/*.csv` and `analyse_summary.json` | Derived experiment summaries; link winners to input NPZ cells and report uncertainty/selection caveats. |
| `Demo/reservoir_metrics/kernel_rank_cg/{kernel_rank_results,montage_data,spatial_rank_results}.npz` | Precomputed dimensionality metrics; attach original cube checksum, singular-value thresholds, centering and time axis. |
| `Demo/**/figures/*.png` | Figures generated by script stages; store plot provenance and avoid changing original published outputs. |
| Generated `m_z` cubes and sample folders | Large raw simulator outputs, not included as full cubes; stream/cache them with run manifests and provide a traceable regeneration workflow. |

### Research and implementation priorities (recommendations only)

1. **Integrity:** verify simulator parameter snapshots, output frame counts, coordinate ordering, six-input robotic provenance and dataset/feature row alignment.
2. **Predictive validity:** put supervised MI fitting, feature selection and model selection inside training folds; compare leakage-free nested evaluation to the current benchmark.
3. **Physics:** independently vary bias field, wave frequency, anisotropy amplitude, damping, pulse separation, mesh resolution and coarse-graining with fixed evaluation splits.
4. **Benchmark fairness:** compare stronger Fourier-feature/neural baselines with equal tuning budgets and measure readout cost separately from physical device drive, sensing and simulation costs.
5. **Evidence:** label results as cache-based figure reproduction, raw simulation reproduction or independent physical device measurement. These levels are not interchangeable.

**Modification boundary:** this branch changes only the root `README.md`; all 26 existing executable files and all other repository artifacts remain unchanged.
