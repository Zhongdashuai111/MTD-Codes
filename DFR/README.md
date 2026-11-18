# Reproducing the Kyber-1024 DFR comparison

This package recomputes the four DFR entries in the subsection
**Application to Kyber and DFR Comparison**. The fixed parameters are
`q=3329`, `n=256`, `k=4`, `eta1=eta2=2`, `du=11`, and `dv=5`.
All ties are counted as errors. The probabilities use a discrete noise model
with dependent coordinates and independent compression errors induced by
uniform residues. The reported numbers are deterministic numerical evaluations
of mathematical DFR bounds; floating-point and geometric computations are
not certified by interval or exact rational arithmetic.

## 1. The four table cells

| Python entry point | Table column | Dimension | Code parameter | Squared minimum toroidal distance | Expected displayed DFR bound |
|---|---|---:|---|---:|---:|
| `DFR_LiuSakzad_l2_gamma446.py` | Codes from Liu–Sakzad | 2 | gamma=446 | 2967048 | 2^(-184.04) |
| `DFR_LiuSakzad_l4_scale554.py` | Codes from Liu–Sakzad | 4 | scale=554 | 3682992 | 2^(-213.56) |
| `DFR_ThisWork_l2_gamma446.py` | This work | 2 | gamma=446 | 2967812 | 2^(-184.15) |
| `DFR_ThisWork_l4_gamma554.py` | This work | 4 | gamma=554 | 3689644 | 2^(-213.72) |

The reference values are recomputed for these fixed codebooks. They are not
transcribed from Liu–Sakzad's published DFR table. The scripts generate the
codebooks and Voronoi constraints from their definitions; they do not read
stored DFR answers or precomputed noise distributions.

`reproduce_all_DFR.py` recomputes all four cells, sharing the 2D joint noise
distribution between the two 2D codebooks. Each separate script prints and
saves the result for the cell named in its filename.

## 2. Installation and execution

The 4D calculations are implemented in Python with NumPy and SciPy and can
run in native Windows Python. The 2D Python entry points build and invoke the
included C/MPC/MPFR numerical backend. This follows the multiprecision 2D FFT
approach in Paiva's Section 6.2; the 2D implementation is not pure Python.
Use Linux or Windows Subsystem for Linux (WSL) to reproduce all four cells.

### Linux or Windows WSL: all four cells

Install the development packages once, inside Ubuntu/Debian or WSL Ubuntu:

```
sudo apt-get update
sudo apt-get install build-essential libgmp-dev libmpfr-dev libmpc-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_all_DFR.py
```

Run the commands from the extracted `Kyber_DFR_Reproduction` directory.
GCC compilation is automatic on the first 2D run. No precompiled executable
is required. The source is bundled; no GitHub clone is performed at runtime.

### Native Windows: the two 4D cells

```
py -3.12 -m pip install -r requirements.txt
py -3.12 DFR_LiuSakzad_l4_scale554.py
py -3.12 DFR_ThisWork_l4_gamma554.py
```

The native Windows commands cover the 4D entries. Run the 2D entries inside
WSL, with the Linux Python interpreter and development libraries installed
there. A Python environment installed on Windows is separate from the WSL
Python environment.

### Individual cells and numerical controls

```
python DFR_LiuSakzad_l2_gamma446.py
python DFR_ThisWork_l2_gamma446.py
python DFR_LiuSakzad_l4_scale554.py
python DFR_ThisWork_l4_gamma554.py
python reproduce_all_DFR.py --precision 260 --threads 8 --output-dir results/check_260
```

The default 2D precision is 320 bits and the default OpenMP thread count is
at most 6. `--precision` and `--threads` affect the 2D backend only. The 4D
FFT uses double precision after exponential tilting. Runs are sequential to
limit memory use. Allow several GiB of available memory for the 2D FFT;
an 8 GiB or larger Linux/WSL memory allowance is recommended.

## 3. Codebooks and distance verification

All definitions appear in `dfr_core/codebooks.py`. Every run checks the code
cardinality and enumerates all distinct pairs to verify the squared minimum
toroidal distance.

For dimension 2, with `a=1664` and `g=446`, the codebooks are:

```
This work:  (0,0), (g,a), (a,q-g), (a+g,a-g).
Reference:  (0,0), (g,a), (a,g),   (a+g,a+g).
```

For dimension 4, the proposed code uses all binary row messages `m` and the
mapping `m @ G mod q`, where `a=1664`, `g=554`, and

```
G = [[a,   g,   g,   g],
     [q-g, a,   g,   q-g],
     [q-g, q-g, a,   g],
     [q-g, g,   q-g, a]].
```

The reference code is the 16-point set in Liu–Sakzad, Example 3, multiplied
by 554. Its 16 unscaled vectors are explicitly listed in `codebooks.py`.
The mapping direction for the proposed matrix is a row-message convention.

## 4. Noise distribution

Let `CBD2` have masses `[1,4,6,4,1]/16` on `[-2,-1,0,1,2]`. The compression
error distribution is computed by enumerating all 3329 uniformly distributed
residues, applying compression and decompression, and centering the difference.
No uniform-interval approximation is substituted for this distribution.
For `du=11`, the masses at `-1,0,1` are respectively `640/3329`, `2048/3329`,
and `641/3329`; this small asymmetry is retained.

For a block of length `ell`, the integer-valued noise is a sum of:

1. `1024/ell` independent negacyclic products of two length-`ell` CBD2 vectors;
2. `1024/ell` independent negacyclic products with coefficient distributions
   CBD2 and `CBD2 + compression_error_u`;
3. a vector whose independent coordinates have distribution
   `CBD2 + compression_error_v`.

Products are computed in the integer ring `Z[x]/(x^ell+1)` before reduction
modulo 3329. In each product the output coordinates are dependent. The
algorithms retain that dependence; only the distinct vector summands in the
above decomposition are independent. The model assumes compression errors
are independent of the other primitive noise sources.

## 5. Common DFR bound and periodic geometry

We use the periodic point set `C + q Z^ell`. For each main codeword `v` in
`C`, collect its neighboring periodic points `v'`, and put `h=v'-v`. A
pairwise error halfspace is

```
<N,h> >= ||h||_2^2 / 2.
```

Its probability is denoted `p_error(v',v)`. Summing these probabilities over
the neighbors and main codewords gives the whole-message bound

```
B = (n/ell)/|C| * sum_{v in C} sum_{v' in N(v)} p_error(v',v).
DFR <= B.
```

Thus the outer factor is 32 in dimension 2 and 4 in dimension 4. Boundaries
are included. Independence between message blocks is not required for the
union bound. The calculation is not an estimate of the exact full-message
DFR.

The periodic copies are essential: a single ordinary difference between two
representatives in `Z_q^ell` does not describe the entire toroidal comparison
event. The squared threshold is the Euclidean norm of the chosen periodic
displacement. `kyber_dfr_subsection.tex` uses this periodic-neighbor version
of the pairwise formula. The same-codeword periodic copies, if present among
the constraints, can be counted conservatively.

`geometry.py` constructs the points with shifts in `{-1,0,1}^ell` and calls
SciPy/Qhull. For every main codeword it checks that the selected halfspaces
define a bounded cell contained in the coordinate cube `[-q/2,q/2]^ell`,
and checks all vertices against all candidate point constraints. Any more
distant periodic copy is coordinatewise dominated, inside that cube, by a
copy among the generated shifts. These are floating-point coverage checks,
not exact rational certificates. Redundant constraints returned by Qhull
are retained; they preserve the upper-bound interpretation.

Equal displacement vectors are accumulated with their exact integer
multiplicities. Distinct directions are not merged using an assumed circular
shift symmetry of the noise. Direction and multiplicity data are saved for
inspection on every run.

## 6. Two-dimensional calculation: Paiva Section 6.2

The following computation is performed by the source in
`dfr_core/native/fft2d_custom.c`, called by the Python entry points.

1. Enumerate the two base bivariate product distributions.
2. Compute their multiprecision 2D FFTs and raise each transform to the power
   512, representing 512 independent copies of each product.
3. Multiply the two transforms and the two coordinate transforms for
   `CBD2 + compression_error_v`.
4. Apply the inverse 2D FFT to obtain the joint distribution on a
   `2048 x 2048` cyclic grid, read through centered integer representatives.
5. For every grid point, count the periodic-neighbor halfspaces it satisfies,
   with their multiplicities. Sum its probability mass times this count and
   multiply by `128/4=32`.

The grid period 2048 is a numerical convolution parameter, not the
cryptographic modulus 3329. A marginal Chernoff calculation bounds the
probability that any of the 256 coefficients has absolute value at least
1024. Multiplying this bound by the average number of counted neighbors
controls the effect of grid folding on the pairwise sum. This correction is
added to the reported numerical value. It is negligible at the displayed
precision; see `log2_folding_correction` in the generated JSON.

For comparison, the backend also calculates the direct toroidal block-error
event and the toroidal pairwise sum over the other three main codewords.
These diagnostics agree with the periodic-neighbor result to the displayed
two decimal places for the present two codebooks. The manuscript and the
primary program output use the periodic-neighbor bound, matching Section 5.

The backend uses Paiva's MPC/MPFR FFT implementation. A run at 260-bit
precision can be compared with the default 320-bit run. Ordinary unshifted
double-precision inverse FFT is inadequate for directly resolving
probability masses of order `2^(-184)`; changing only the NumPy FFT size
does not provide the needed precision.

## 7. Four-dimensional calculation: Paiva Section 6.3, Algorithm 3

The Python implementation in `projection4d.py` follows the projection and
Voronoi-neighbor framework of Algorithm 3. It independently implements the
numerical convolution using exponential tilting. Paiva's own implementation
instead uses arbitrary-precision FFTs, including the Bailey transform
described in Section 6.3. This package does not claim to reproduce that
numerical implementation line for line.

For each integer displacement `h`:

1. Divide its coordinates by their positive greatest common divisor `g`.
   The integer projection threshold is `ceil(||h||^2/(2g))`.
2. Enumerate the 625 possible values of the first CBD2 vector. The second
   vector has 625 possibilities for the first product distribution and 2401
   possibilities for the second. For `A,B` and a primitive direction `u`,
   the scalar product is `sum A_i M_ij B_j`, where
   `M_ij = u[(i+j) mod 4]` with a minus sign when `i+j >= 4`.
3. Form the two discrete one-dimensional base distributions, repeated 256
   times each, and the four final coordinate-noise projections.
4. Compute their combined cumulant generating function `K(t)`. Choose the
   positive tilting parameter with `K'(t)` equal to the tail threshold `T`.
5. Tilt each atom by `exp(t*x)`, normalize it, and convolve the tilted
   distributions by real FFT. Recover the original tail from

```
Pr(S >= T) = exp(K(t)-t*T) * sum_{s>=T} Pr_t(S=s) exp(-t*(s-T)).
```

This is a deterministic distribution identity; no random samples are drawn.
The tilted distribution places substantial mass around the threshold,
making the tiny original tail accessible to double-precision convolution.

The retained tail window has width `ceil(48/t)`. The FFT size accounts for
the tilted variance, this window, and the distance between the tilted mean
and the threshold. The program records Chernoff controls for aliasing and
the exponentially weighted truncation remainder. Selected directions are
also recomputed with a changed tilt and a larger FFT grid.

If a direction's Chernoff log-bound is more than 65 bits below the largest
directional Chernoff log-bound, its entire Chernoff bound is included as a
remainder instead of calculating its FFT tail. These directions are not
discarded. Finally, directional tails are weighted by their multiplicities,
summed stably in logarithmic form, and multiplied by `64/16=4`.

## 8. Output and verification

Every run creates the directory selected by `--output-dir`. Defaults are
`results/all/` for all four cells and `results/<case>/` for an individual cell.
The directory contains:

- `DFR_<case>.json`: parameters, distance, unrounded DFR exponent, method,
  numerical checks, and the table entry;
- `geometry_<case>.json`: explicitly generated codewords, directions,
  multiplicities, and geometric verification results;
- `directions_<4D case>.json`: all 4D directional tails and Chernoff remainders;
- `fft2d_<precision>_raw.json` and `fft2d_backend.log`: 2D raw results and log;
- `summary.csv`: one row per requested table cell;
- `environment.json`: actual software versions, platform, thread setting,
  and total run time.

`verified_run/` contains the outputs of an actual run of the distributed
program. The program does not use this folder as input. Run-time metadata
there describe the verification environment, not the user's computer.

The constants in the table above are comparison targets only. Small changes
in the last digits can arise from libraries, thread counts, floating-point
summation order, or redundant Voronoi constraints. Compare the two-decimal
table exponents. Rounding a bound estimate to two decimal places is not a
directed rounding operation that turns it into a certified upper bound.

## 9. References and source attribution

1. T. B. Paiva, M. A. Simplicio Jr., S. M. Hafiz, B. Yildiz,
   E. L. Cominetti, and H. S. Ogawa, *Tailorable codes for lattice-based KEMs
   with applications to compact ML-KEM instantiations*, ePrint 2024/1243,
   revision dated 23 April 2025, also published in TCHES 2025.
   <https://eprint.iacr.org/2024/1243>
2. Author implementation: <https://github.com/thalespaiva/minal_mlkem>,
   commit `6b12ae4fa24c04fe756386cb8d7e7d88d283d1a9`.
3. Liu–Sakzad, the manuscript reference `LiuSakzad2026`:
   <https://arxiv.org/abs/2601.08452>.

The section and algorithm numbers refer to the 25-page revised Paiva paper:
Section 6.2 is the 2D procedure; Section 6.3 and Algorithm 3 are the 4D
procedure. Algorithm 3 is not a 2D algorithm in this version. Earlier drafts
have different algorithm numbering.

The Paiva source files are retained under their Apache-2.0 license in
`dfr_core/native/vendor/LICENSE`. The adapter changes the codebooks,
decoding-region summation, grid configuration, and output format. See
`dfr_core/native/NOTICE.md`. The 4D probability evaluator and the Python
entry points are separate implementations prepared for this comparison.

## 10. Recorded reproduction run

The bundled verification run completed all four cells in approximately 209.2 seconds on the recorded Linux environment, using Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and six OpenMP threads for the 2D backend. It used the NumPy 4D implementation without a compiled projection extension. These timings are not predictions for a different computer.

| Cell | Unrounded log2 of the evaluated bound |
|---|---:|
| `LiuSakzad_l2_gamma446` | -184.037888446259 |
| `ThisWork_l2_gamma446` | -184.154976325681 |
| `LiuSakzad_l4_scale554` | -213.555736778437 |
| `ThisWork_l4_gamma554` | -213.723909128836 |

The two 2D exponents agreed at the 14 printed decimal places between 260-bit and 320-bit runs. Selected 4D tail checks differed by less than 3e-13 in their log2 values when changing the tilt and FFT grid. The source-cleanup checks also reproduced the selected tails. Full details are in `verified_run/additional_checks.json`.

`requirements-verified.txt` records the exact NumPy and SciPy versions used. The broader `requirements.txt` is suitable for ordinary installation.
