"""Python entry points for the four table cells; all probabilities are recomputed."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from .codebooks import CASES, PARAMETERS, codebook
from .geometry import periodic_neighbors
from .noise_model import compression

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT/'dfr_core'/'native'

def log2_folding_bound(average_neighbors):
    """Chernoff bound for the whole-message effect of a 2048-periodic 2D grid."""
    cbd = np.array([1,4,6,4,1],dtype=float)/16
    pu,ou = compression(11); pv,ov = compression(5)
    parts = []
    for p,lo in ((cbd,-2),(np.convolve(cbd,pu),ou-2)):
        values = (np.arange(-2,3)[:,None]*(np.arange(len(p))+lo)).ravel()
        weights = (cbd[:,None]*p).ravel()
        parts.append((weights,values,1024))
    e = np.convolve(cbd,pv)
    parts.append((e,np.arange(len(e))+ov-2,1))
    tails = []
    for sign in (-1,1):
        def objective(t):
            return sum(r*logsumexp(np.log(p)+t*sign*x) for p,x,r in parts)-1024*t
        r = minimize_scalar(objective,bounds=(0,1),method='bounded',
                            options={'xatol':1e-14})
        tails.append(float(r.fun))
    return (math.log(256*average_neighbors)+logsumexp(tails))/math.log(2)

def fft2d(geometries, output, bits, threads):
    if os.name == 'nt':
        raise RuntimeError('The 2D MPC/MPFR backend requires Linux or Windows WSL. See README.md. The 4D scripts run in native Windows Python.')
    cc = shutil.which(os.environ.get('CC','gcc'))
    if not cc:
        raise RuntimeError('GCC is missing. Install build-essential, libgmp-dev, libmpfr-dev and libmpc-dev; see README.md.')
    build = ROOT/'build'; build.mkdir(exist_ok=True)
    exe = build/f'fft2d_{bits}_2048'
    sources = [NATIVE/'fft2d_custom.c',*(NATIVE/'vendor'/f for f in
               ['fft_mpc.c','mp_matrix.c','dfr_2d_codes.c'])]
    tracked = [*sources,*list((NATIVE/'vendor').glob('*.[ch]'))]
    if not exe.exists() or exe.stat().st_mtime < max(p.stat().st_mtime for p in tracked):
        command = [cc,'-O3','-fopenmp','-I',str(NATIVE/'vendor'),
                   '-DCOMPUTE_JOINT_DISTRIBUTIONS_AND_DFR',
                   '-DN_SUPPORT_DIST_2D=2048',f'-DN_PRECISION_BITS={bits}',
                   *map(str,sources),'-lmpfr','-lmpc','-lgmp','-lm','-o',str(exe)]
        compiled = subprocess.run(command,capture_output=True,text=True)
        if compiled.returncode:
            raise RuntimeError('Could not compile the MPC/MPFR backend. See README.md for dependencies.\n'+compiled.stderr)
    directions = build/'directions_2d.txt'
    with directions.open('w') as f:
        for case in ('ThisWork_l2_gamma446','LiuSakzad_l2_gamma446'):
            h,m,g = geometries[case]
            print(len(h),file=f)
            for v,weight in zip(h,m):print(*v,int(weight),file=f)
    env = dict(os.environ,OMP_NUM_THREADS=str(threads))
    print(f'Computing 2D joint distribution: {bits} bits, {threads} threads.',flush=True)
    with (output/'fft2d_backend.log').open('w') as log:
        proc = subprocess.run([str(exe),str(directions)],env=env,
                              stdout=subprocess.PIPE,stderr=log,text=True)
    if proc.returncode:
        raise RuntimeError(f'2D FFT backend failed; see {output / "fft2d_backend.log"}.')
    raw = json.loads(proc.stdout)
    (output/f'fft2d_{bits}_raw.json').write_text(json.dumps(raw,indent=2))
    mapping = {'ours_gamma446':'ThisWork_l2_gamma446',
               'previous_gamma446':'LiuSakzad_l2_gamma446'}
    result = {}
    for item in raw['cases']:
        case = mapping[item['name']]
        fold = log2_folding_bound(float(np.sum(geometries[case][1]))/4)
        value = item['log2_voronoi_pairwise']
        combined = float(np.logaddexp(value*math.log(2),fold*math.log(2))/math.log(2))
        result[case] = dict(log2_bound=combined,log2_raw_neighbor_sum=value,
                            log2_folding_correction=fold,precision_bits=bits,
                            joint_grid=[2048,2048],backend_seconds=raw['elapsed_seconds'],
                            log2_direct_toroidal_block_bound=item['log2_ties_error'],
                            log2_toroidal_pairwise_bound=item['log2_pairwise_ties'],
                            negative_mass=raw['negative_mass'],imaginary_l1=raw['imaginary_l1'])
    return result

def execute(selected,args):
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    needed = list(selected)
    if any(CASES[c][0] == 2 for c in selected):
        needed = list(dict.fromkeys(needed+['ThisWork_l2_gamma446','LiuSakzad_l2_gamma446']))
    geometries = {}
    for c in needed:
        geometries[c] = periodic_neighbors(codebook(c))
        h,m,g = geometries[c]
        (output/f'geometry_{c}.json').write_text(json.dumps(dict(codewords=codebook(c).tolist(),
            directions=h.tolist(),multiplicities=m.tolist(),verification=g),indent=2))
    two = fft2d(geometries,output,args.precision,args.threads) if any(CASES[c][0]==2 for c in selected) else {}
    rows = []
    for c in selected:
        before = time.perf_counter()
        ell,family,distance = CASES[c]
        h,m,g = geometries[c]
        if ell == 2:
            detail = two[c]
            algorithm = 'Paiva Section 6.2: multiprecision 2D FFT, periodic-neighbor halfspace summation'
        else:
            from .projection4d import run
            detail = run(c,h,m,output/f'directions_{c}.json')
            detail['log2_bound'] = detail['log2_combined_bound']
            algorithm = 'Paiva Section 6.3, Algorithm 3: 1D projections; independent tilted-FFT implementation'
        value = detail['log2_bound']
        record = dict(case=c,parameters=PARAMETERS,ell=ell,code_size=2**ell,
                      squared_minimum_toroidal_distance=distance,
                      algorithm=algorithm,ties_counted_as_errors=True,
                      log2_bound=value,table_entry=f'2^({value:.2f})',
                      status='deterministic numerical evaluation; not interval-certified',
                      geometry=g,numerical_details=detail,
                      case_seconds=detail.get('backend_seconds',time.perf_counter()-before))
        (output/f'DFR_{c}.json').write_text(json.dumps(record,indent=2))
        row=dict(case=c,ell=ell,dT_squared=distance,log2_DFR_bound=f'{value:.12f}',DFR_bound=record['table_entry'])
        rows.append(row)
        print(f'\n{c}: ell={ell}, |C|={2**ell}, dT^2={distance}\n'
              f'log2(DFR bound) = {value:.12f}\nDFR bound = {record["table_entry"]}\n',flush=True)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    env = dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
               platform=platform.platform(),processor=platform.processor(),
               logical_cpu_count=os.cpu_count(),fft2d_threads=args.threads,
               elapsed_seconds=time.perf_counter()-start)
    (output/'environment.json').write_text(json.dumps(env,indent=2))
    return rows

def main(case=None):
    p=argparse.ArgumentParser(description='Recompute a specified cell, or all four cells, of the Kyber DFR comparison.')
    p.add_argument('--precision',type=int,choices=[260,320],default=320,
                   help='MPC/MPFR precision for the 2D backend; ignored by 4D calculations.')
    p.add_argument('--threads',type=int,default=min(6,os.cpu_count() or 1),
                   help='OpenMP threads for the 2D backend.')
    p.add_argument('--output-dir',default=str(ROOT/'results'/('all' if case is None else case)))
    args=p.parse_args()
    if args.threads<1:p.error('--threads must be positive')
    try:
        execute(list(CASES) if case is None else [case],args)
    except (RuntimeError,ValueError) as e:
        p.exit(1,f'Error: {e}\n')
