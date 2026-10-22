# Spin-wave mathematics prototype (research experiment)

This branch adds a **separate** CPU math reference/verification suite and a seeded benchmark using the repository's actual **cached** spin-wave-reservoir (SWRC) features. It does not modify the existing robotic experiment, MuMax3 driver, or original data.

## Claims and boundaries

- `math_engine/exact.py` solves a restricted subset of K–12/precalculus questions **using SymPy on the CPU**. Its score is not a spin-wave computing achievement.
- `math_engine/benchmark.py --source cached` **trains readouts on existing SWRC feature vectors** (`swrc_features.npz`, variant 2), pairs them with `Train2X.csv` sensor values, and asks whether the field encodes arithmetic relationships of the **first two of six sensor inputs**. This is a genuine cached spin-wave feature-space experiment but not a newly driven two-input magnetic computation. We cannot infer that these wave patterns implement arithmetic directly.
- `--source cube` reads actual **MuMax3-produced** `(N,T,H,W)` cubes from the existing simulator plus its `run_index.json`. It extracts a deterministic, label-independent 12-frame × 8 × 8 coarse-grained representation. Generate cubes with existing `simulator/run_sweep.py` and `simulator/build_cube.py` on a compatible GPU first. A proper full simulator experiment should be run before making arithmetic-hardware claims.
- No fabricated device or real-time hardware tested. The wave regression heads are ordinary CPU Ridge models. Results are exploratory; the exact evaluator is the reliable reference.

## Reproduce

At the repository root with `numpy`, `scikit-learn`, `sympy` installed:

```bash
python -m unittest discover -s tests -v
python -m math_engine.benchmark --seed 20261022 --per-kind 40 --source cached --report math_results.json
# exact CPU only:
python -m math_engine.benchmark --source none --seed 20261022 --report exact_results.json
# for a real simulator-generated cube with its run index:
python -m math_engine.benchmark --source cube --cube /path/mini_cube.npy --index /path/run_index.json --report cube_math.json
```

The benchmark generates `21 × --per-kind` questions from arithmetic through precalculus, with expected answers derived independently of the solver. It reports per-topic correctness and original questions, as well as genuinely measured rounded-exact readout accuracy/MAE for addition, subtraction, multiplication, greater-than comparison and modulo. Two splits: randomized held-out *rows* and held-out *ordered operand pairs* (harder generalization). Feature/channel selection, scaling and model fitting occur on training samples only. The same splits compare wave features, a raw six-coordinate linear model, and a quadratic polynomial raw-coordinate model.

**Interpretation:** Multiplication of the first two inputs is trivially representable by second-degree polynomial features of the raw coordinates. A useful wave-based computing result would need to beat strong, fairly tuned baselines and report full measurement/simulation/energy costs. Do not conflate the exact SymPy solution suite with physical-device performance.

## Current limits

Expressions are intentionally constrained to generated formats and an AST allowlist; this is not a general natural-language model. Trigonometric special angles, algebraic equations, rational simplification, geometry formulas and linear systems are covered. Solving arbitrary text, recovering excluded-domain information after rational cancellation, proof generation, physical noise robustness and unrestricted symbolic problem solving are not yet covered.

## Files

- `math_engine/questions.py`: seeded question generator and independently constructed ground truth.
- `math_engine/exact.py`: restricted AST parser, symbolic reference solver, output normalization.
- `math_engine/benchmark.py`: exact test run, cached physical feature readouts, MuMax3 cube loader, strict group splits, metrics, JSON report.
- `tests/test_math_engine.py`: reproducible seeded, safety and data-contract regression tests.
