"""Seeded math benchmark: exact SymPy CPU oracle and genuine cached SWRC readouts.

IMPORTANT: exact QA never calls the magnetic reservoir. SWRC benchmark uses
published cached features, not a fresh MuMax3 calculation or fabricated chip.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import time
import sys
from collections import Counter
import numpy as np
import sympy as sp
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from .exact import solve, solution_key
from .questions import questions

TASKS={
    'addition': lambda a,b: a+b,
    'subtraction':lambda a,b: a-b,
    'multiplication':lambda a,b:a*b,
    'comparison_gt':lambda a,b:(a>b).astype(np.float64),
    'modulo':lambda a,b:a % np.where(b == 0, 1, b),
}


def exact_benchmark(seed: int, per_kind: int) -> dict:
    qs=questions(seed,per_kind)
    rows=[]; wrong=[]
    for q in qs:
        try:
            predicted=solve(q.prompt)
            ok=solution_key(predicted)==solution_key(q.expected)
            error=None
        except Exception as e:
            predicted=None;ok=False;error=f'{type(e).__name__}: {e}'
        rows.append({'kind':q.kind,'question':q.prompt,'expected':str(q.expected),
                     'predicted':str(predicted),'correct':bool(ok),**({'error':error} if error else {})})
        if not ok:wrong.append(rows[-1])
    counts=Counter(q.kind for q in qs)
    good=Counter(r['kind'] for r in rows if r['correct'])
    return {'seed':seed,'n_questions':len(qs),'n_correct':sum(r['correct'] for r in rows),
            'accuracy':sum(r['correct'] for r in rows)/len(rows),
            'per_kind':{k:{'correct':good[k],'total':v,'accuracy':good[k]/v} for k,v in sorted(counts.items())},
            'first_questions':rows[:20], 'errors':wrong[:100],
            'note':'CPU/SymPy exact reference, NOT a wave-computing accuracy measurement.'}


def physical_readout(features, coords, seed=20261022, max_voxels=128):
    """Train-only preprocessing and two evaluation splits, including unseen operand pairs.

    Features are already z-scored in the cached repository .npz.  Do not
    use the other four sensors as a label source.  Compare spin features
    against raw linear and polynomial baselines on identical train/test IDs.
    """
    X=np.asarray(features,dtype=np.float32)
    coord=np.asarray(coords,dtype=np.float64)
    if X.ndim!=2 or coord.ndim!=2 or len(X)!=len(coord) or coord.shape[1]<2:
        raise ValueError('expected matching samples and at least two numeric inputs')
    a,b=coord[:,0],coord[:,1]
    groups=np.array([f'{int(aa)},{int(bb)}' for aa,bb in zip(a,b)])
    idx=np.arange(len(X))
    rtrain,rtest=train_test_split(idx,test_size=.25,random_state=seed)
    gtrain,gtest=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(idx,groups=groups))
    reports={}
    for split_name,(tr,te) in {'random_rows':(rtrain,rtest),'unseen_operand_pairs':(gtrain,gtest)}.items():
        if len(set(groups[tr])&set(groups[te])) and split_name=='unseen_operand_pairs':
            raise AssertionError('group leakage')
        # Select high-variance wave channels using TRAIN samples only.
        rank=np.argsort(np.var(X[tr],axis=0,dtype=np.float64))[-min(max_voxels,X.shape[1]):]
        physical=X[:,rank]
        scores={}
        for task,fn in TASKS.items():
            y=fn(a,b).astype(np.float64)
            # The native MuMax3 coordinate grid includes zero: modulo by zero
            # is undefined, so exclude those rows from training AND scoring.
            task_tr=tr[b[tr] != 0] if task=='modulo' else tr
            task_te=te[b[te] != 0] if task=='modulo' else te
            if len(task_tr)<2 or len(task_te)==0:
                raise ValueError(f'insufficient valid rows for {task} on {split_name}')
            predictions={}
            for name,features_,degree,alpha in (
                    ('wave_128',physical,1,10.0),
                    ('raw_linear',coord,1,1e-7),
                    ('raw_polynomial_2',coord,2,1e-7)):
                steps=([PolynomialFeatures(degree=2,include_bias=False)] if degree==2 else [])
                model=make_pipeline(*steps,StandardScaler(),Ridge(alpha=alpha))
                model.fit(features_[task_tr],y[task_tr])
                yp=model.predict(features_[task_te])
                predictions[name]={'exact_accuracy_rounded':float(np.mean(np.rint(yp)==y[task_te])),
                                   'mae':float(np.mean(np.abs(yp-y[task_te]))),
                                   'test_evaluated':int(len(task_te))}
            scores[task]=predictions
        reports[split_name]={'train_rows':len(tr),'test_rows':len(te),
                             'train_unique_pairs':len(set(groups[tr])),
                             'test_unique_pairs':len(set(groups[te])),
                             'results':scores}
    return reports


def load_cached_wave(repo_root: pathlib.Path):
    data=repo_root/'Demo/2_3_5_Robotic_output_classification_task/data'
    features_file=data/'swrc_features.npz'
    with np.load(features_file,allow_pickle=False) as npz:
        features=npz['psi'][0].copy()
    coords=np.loadtxt(data/'Train2X.csv',delimiter=',',dtype=np.float32)
    if features.shape!=(1000,2500) or coords.shape!=(1000,6):
        raise ValueError(f'unexpected shapes features={features.shape} coords={coords.shape}')
    return features,coords


def load_native_cube(cube_path: pathlib.Path,index_path:pathlib.Path, frames=12, side=8):
    """Extract a small physical-state feature set from run_sweep/build_cube output.

    Requires real MuMax3-produced samples. Frames and spatial averaging are
    deterministic and label-blind; no synthetic replacement is performed.
    """
    cube=np.load(cube_path,mmap_mode='r',allow_pickle=False)
    with open(index_path,encoding='utf-8') as f:index=json.load(f)
    if cube.ndim!=4 or cube.shape[2]!=cube.shape[3]:
        raise ValueError('expected N,T,H,W square-field cube')
    n,t,h,w=cube.shape
    if len(index)!=n or h<side or h%side!=0 or w%side!=0:
        raise ValueError('cube/index mismatch or invalid spatial downsampling')
    indices=np.unique(np.linspace(0,t-1,min(frames,t)).round().astype(int))
    out=np.empty((n,len(indices)*side*side),dtype=np.float32)
    coords=np.empty((n,2),dtype=np.float32)
    block=h//side
    for i in range(n):
        record=index[str(i)]
        coords[i]=[record['X1'],record['X2']]
        out[i]=np.asarray(cube[i,indices],dtype=np.float32).reshape(len(indices),side,block,side,block).mean(axis=(2,4)).ravel()
    return out,coords


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=20261022)
    ap.add_argument('--per-kind',type=int,default=40)
    ap.add_argument('--source',choices=['cached','cube','none'],default='cached')
    ap.add_argument('--repo-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
    ap.add_argument('--cube',type=pathlib.Path)
    ap.add_argument('--index',type=pathlib.Path)
    ap.add_argument('--report',type=pathlib.Path,default=pathlib.Path('math_results.json'))
    args=ap.parse_args(argv)
    tic=time.perf_counter()
    exact=exact_benchmark(args.seed,args.per_kind)
    wave=None
    if args.source=='cached':
        features,coords=load_cached_wave(args.repo_root)
        wave=physical_readout(features,coords,seed=args.seed)
        provenance='Repository supplied SWRC 2500-channel frame-100 feature cache, class-2 task (six sensors); first two sensor distances relabeled as arithmetic operands 1..6.'
    elif args.source=='cube':
        if not args.cube or not args.index:
            ap.error('--source cube requires --cube and --index')
        features,coords=load_native_cube(args.cube,args.index)
        wave=physical_readout(features,coords,seed=args.seed)
        provenance='User supplied cube + run_index; no MuMax3 simulation run by this benchmark.'
    else:
        provenance='Exact CPU baseline only; wave benchmark not executed.'
    report={'experiment':'Spin-wave mathematics prototype v0.1','seed':args.seed,
            'exact_reference':exact,'wave_readout':wave,'wave_provenance':provenance,
            'physical_hardware_tested':False,'fresh_mumax_simulation':False,
            'elapsed_seconds':round(time.perf_counter()-tic,3)}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with open(args.report,'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(f"[exact CPU] seed={args.seed}  correct={exact['n_correct']}/{exact['n_questions']}  ({100*exact['accuracy']:.1f}%)")
    if wave:
        for split,s in wave.items():
            print(f'[cached SWRC] {split}: {s["train_rows"]} train / {s["test_rows"]} test')
            for task,v in s['results'].items():
                print(f'  {task:15} wave={100*v["wave_128"]["exact_accuracy_rounded"]:5.1f}%'
                      f'  raw-linear={100*v["raw_linear"]["exact_accuracy_rounded"]:5.1f}%'
                      f'  raw-quadratic={100*v["raw_polynomial_2"]["exact_accuracy_rounded"]:5.1f}%')
    print(f'[report] {args.report}')
    return 0 if exact['n_correct']==exact['n_questions'] else 1

if __name__=='__main__':sys.exit(main())
