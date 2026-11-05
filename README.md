# Computational Results for Maximum Toroidal Distance Codes

This repository contains Python programs supporting the computational results in the manuscript **“Bounds and Constructions of Maximum Toroidal Distance Codes”** by **Pengjie Zhong, Jinquan Luo, and Yufeng Song**.

The programs cover the local ball--Plotkin bound, Delsarte linear programming (LP) bounds, and exact graph-based searches for maximum toroidal distance (MTD) codes. They are intended to facilitate independent reproduction of the numerical results in the manuscript.

## Mathematical setting

For vectors $x,y\in\mathbb Z_q^n$, the toroidal distance is

$$
d_T(x,y)=\left(\sum_{i=1}^n\min\{|x_i-y_i|,\;q-|x_i-y_i|\}^2\right)^{1/2}.
$$

An MTD code maximizes the minimum toroidal distance among codes $C\subseteq\mathbb Z_q^n$ with $|C|=2^n$. The LP and local Plotkin scripts report upper bounds on the **squared** minimum distance, $d_T(C)^2$.

## Repository contents

| File | Corresponding result in the manuscript | Purpose |
| --- | --- | --- |
| `Example 2_local_plotkin_bound(1).py` | Example 2 | Computes the local ball--Plotkin bound for $n=2$, $q=4$, and $M=4$ using exact integer and rational arithmetic. |
| `Example 3_Delsarte LP Bound(2).py` | Example 3 | Computes $L_4(8,\delta)$ for the example involving $2E_8\cap\mathbb Z_4^8$ ($M=256$). |
| `Example 5_Delsarte LP Bound.py` | Example 5 | Computes $L_6(4,\delta)$ for $M=16$. |
| `Example 6_Delsarte LP Bound .py` | Example 6 | Computes $L_{12}(4,\delta)$ for $M=16$. |
| `Example 7.zip` | Example 7 | Contains the $n=16$, $q=4$ LP script and distance-checking scripts for binary-message subsets of specific generator matrices. See the scope note below. |
| `Example_7_full_Z4_code_distance.py` | Example 7 | Exhaustively checks all $4^8-1=65,535$ nonzero codewords of the full $\mathbb Z_4$-linear code given in the manuscript, obtaining $d_T(C)^2=9$. |
| `Table IV.zip` | Table IV | Contains 12 Delsarte LP scripts and 12 local ball--Plotkin scripts, one of each for every parameter pair in the comparison table. |
| `Table V MTD_exhaustive_search.py` | Table V | Implements an exact graph-based search for MTD codes, subject to a time limit. |

The two ZIP archives must be extracted before their contained Python programs can be run. Within `Table IV.zip`, the `Delsarte_LP_Bound_files` and `Local_plotkin_bound` directories contain scripts labeled by their $(n,q)$ parameters.

## Software requirements

- Python 3.12 (the version used for the manuscript's computations).
- NumPy and SciPy for the Delsarte LP scripts (`pip install numpy scipy`).
- The LP scripts use `scipy.optimize.linprog` with the HiGHS solver; the primal and dual feasibility tolerances are set to $10^{-9}$ by default.
- The local ball--Plotkin, graph-search, and distance-checking scripts use only the Python standard library.

The manuscript reports LP computations using Python 3.12.0, SciPy 1.17.1, and HiGHS 1.12.0 on an AMD Ryzen 7 7735U computer with 16 GB RAM. Execution times can vary by machine and software version.

## Running the programs

Run the files from a terminal in the directory where they are located. Because some file names contain spaces, enclose the file names in quotation marks.

**Local ball--Plotkin bound (Example 2):**

```bash
python "Example 2_local_plotkin_bound(1).py"
```

The parameters `n`, `q`, and `M` can be edited near the beginning of this script. Its ball-volume calculations and resulting local bound use exact integer/rational arithmetic.

**Delsarte LP bound (Example 3):**

```bash
python "Example 3_Delsarte LP Bound(2).py"
```

For a specified squared-distance threshold, the LP scripts also accept command-line arguments such as:

```bash
python "Example 3_Delsarte LP Bound(2).py" --n 8 --q 4 --M 256 --delta 9
```

Without `--delta`, the scripts search for the first threshold $\delta_0$ for which $L_q(n,\delta_0)<M$ and display nearby LP values. The resulting integer upper bound is $d_T(C)^2\le\delta_0-1$.

**Full-code distance check (Example 7):**

```bash
python "Example_7_full_Z4_code_distance.py"
```

This program enumerates all $65,535$ nonzero codewords of the $\mathbb Z_4$-linear code, returning $d_T(C)^2=9$ and $d_T(C)=3$.

**Exact search (Table V):**

```bash
python "Table V MTD_exhaustive_search.py" 2 5
```

The two positional arguments are the code length $n$ and the modulus $q$. A completed search determines the largest attainable minimum squared toroidal distance for a code of size $2^n$ and returns a corresponding codebook. The program uses a compatibility graph, a bitset branch-and-bound clique algorithm, greedy-coloring upper bounds, and binary search over attainable squared distances. Its default time limit is five days (120 hours). A run that reaches the time limit must **not** be interpreted as an optimality proof.

## Reproducing the reported results

- **Example 2:** The local ball--Plotkin calculation for $(n,q,M)=(2,4,4)$ gives $d_T(C)^2\le4$.
- **Example 3:** For $(n,q,M)=(8,4,256)$, the LP gives the exclusion threshold $\delta_0=9$, yielding $d_T(C)^2\le8$.
- **Example 5:** For $(n,q,M)=(4,6,16)$, the LP gives $\delta_0=13$, yielding $d_T(C)^2\le12$.
- **Example 6:** For $(n,q,M)=(4,12,16)$, the LP gives $\delta_0=49$, yielding $d_T(C)^2\le48$.
- **Example 7:** For $(n,q,M)=(16,4,65536)$, the LP gives $\delta_0=13$, yielding $d_T(C)^2\le12$. Independent exhaustive enumeration of the full constructed code gives $d_T(C)^2=9$.
- **Table IV:** The archive contains the local ball--Plotkin and Delsarte LP calculations for all 12 parameter pairs in the table.
- **Table V:** The graph-search script can be run with each parameter pair listed in the table. Reported runtimes are machine-dependent; searches that time out do not establish optimality.

**Scope of the Example 7 distance scripts.** The manuscript's Example 7 defines a $\mathbb Z_4$-linear code generated by $G_0$, containing $4^8=65,536$ codewords. The two distance-checking scripts currently in `Example 7.zip` instead enumerate codewords obtained from **binary** messages $m\in\{0,1\}^8$, giving only $2^8=256$ codewords for their respective matrices. Their reported distances therefore apply to those binary-message subsets, not automatically to the full $\mathbb Z_4$-linear codes. The separate `Example_7_full_Z4_code_distance.py` file remedies this limitation for the manuscript's generator $G_0$ by enumerating all messages in $\mathbb Z_4^8$. The ZIP archive also contains a separate LP script computing the reported upper bound.

**Numerical precision.** The LP calculations are numerical unless an exact primal/dual certificate is successfully produced and verified. A value appearing close to an integer is not, by itself, an exact optimality certificate. Some exact-certificate modes are available only for supported moduli and problem sizes.

## Citation

If you use these programs, please cite the associated manuscript:

Pengjie Zhong, Jinquan Luo, and Yufeng Song, *Bounds and Constructions of Maximum Toroidal Distance Codes*, manuscript.

For an immutable version of these computational materials, refer to a tagged GitHub release or a specific commit of this repository.
