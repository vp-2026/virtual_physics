#!/usr/bin/env python3
"""Offline random-placement and no-motion references for the finalized release."""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import numpy as np
from scipy.stats import hypergeom

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulator/runtime'))
from pyGameWorld import loadFromDict
from analyze_action_predictions import score_attempt
from record_scoring import recorded_attempt

MODELS=('gpt5','gemini31pro','qwen36plus','robotics_er2')
SEED=20261115
DRAWS=10000

def read(path): return json.loads(path.read_text())
def write(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def close(a,b,label):
    assert np.allclose(a,b,atol=1e-10,rtol=0),(label,a,b)

def interval(rows,key,weight=None,seed=SEED,draws=DRAWS):
    """Pointwise whole-layout bootstrap; keep all tasks/directions/attempts together."""
    by=collections.defaultdict(list)
    for row in rows: by[row['layout']].append(row)
    labels=sorted(by)
    num=np.asarray([sum(float(r[key])*float(r[weight] if weight else 1) for r in by[l]) for l in labels])
    den=np.asarray([sum(float(r[weight] if weight else 1) for r in by[l]) for l in labels])
    samples=np.random.default_rng(seed).integers(len(labels),size=(draws,len(labels)))
    values=num[samples].sum(axis=1)/den[samples].sum(axis=1)
    return dict(estimate=float(num.sum()/den.sum()),ci95=np.quantile(values,[.025,.975]).tolist(),
                n=len(rows),tasks=len({r['goal'] for r in rows}),layouts=len(labels),
                bootstrap_draws=draws,bootstrap_seed=seed)

def pq_no_motion(start,truth):
    # The baseline predicts that an initially visible object stays in-scene.
    if truth['state']!='in_scene': return 0.0
    return math.exp(-sum((a-b)**2 for a,b in zip(start,truth['point']))/(2*150.0**2))

def endpoint(spec):
    x,y=float(spec['final_x']),float(spec['final_y'])
    return {'state':'in_scene' if 0<=x<=599 and 0<=y<=599 else 'exited','point':[x,y]}

def uniform_search(p,attempts):
    if p==1: return 1.0
    return -math.expm1(attempts*math.log1p(-p))

def distinct_search(k,n,attempts):
    if n-k<attempts: return 1.0
    return -math.expm1(sum(math.log1p(-k/(n-j)) for j in range(attempts)))

def csv_out(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)

def analyze_baselines(repo,results,out):
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new/empty baseline output directory')
    out.mkdir(parents=True,exist_ok=True)
    tasks={g['balanced_goal_id']:g for g in read(repo/'benchmark/tasks.json')['goals']}
    pops=read(results/'populations.json'); original=read(results/'summary.json')
    main=set(pops['main']); transfer=set(pops['matched_transfer'])
    er_main=set(pops['robotics_main']); er_transfer=set(pops['robotics_transfer'])
    manifest=read(repo/'data/finalized/manifest.json')
    initial={}
    for f in sorted((repo/'benchmark/assets').glob('*/world.json')):
        world=loadFromDict(read(f)['world'])
        initial[f.parent.name]={name:[float(x) for x in obj.getPos()]
                                for name,obj in world.objects.items() if not name.startswith('_') and not obj.isStatic()}
        assert all(0<=v<=599 for xy in initial[f.parent.name].values() for v in xy)

    density=[]
    for gid,g in tasks.items():
        if g.get('p_success') is None: continue
        k=g.get('event_count') if g.get('event_count') is not None else g['count']
        n=g.get('valid_placement_count') if g.get('valid_placement_count') is not None else g['valid_placements']
        assert int(k)==k and int(n)==n and 0<=k<=n and n>=8
        close(k/n,g['p_success'],gid+' density/count ratio')
        density.append(dict(goal=gid,layout=g['run_name'].rsplit('_',1)[0],in_main=int(gid in main),
                            successful_grid_placements=int(k),valid_grid_placements=int(n),
                            p1=float(k/n),p8=uniform_search(k/n,8),p8_distinct=distinct_search(k,n,8)))
        close(density[-1]['p8_distinct'],hypergeom.sf(0,int(n),int(k),8),gid+' distinct-search probability')
    assert len(density)==1560 and sum(r['in_main'] for r in density)==1509
    density_by={r['goal']:r for r in density}
    attempts={m:[] for m in MODELS}; probes={m:[] for m in MODELS}; solves={m:[] for m in MODELS}
    validated=0
    for shard in manifest['shards']:
        m=shard['model']; ids=er_main if m=='robotics_er2' else main
        novel_ids=er_transfer if m=='robotics_er2' else transfer
        with gzip.open(repo/'data/finalized'/shard['path'],'rt') as f:
            for line in f:
                r=json.loads(line); summary=r['summary']; goal=summary['goal']; gid=goal['balanced_goal_id']
                if gid not in ids: continue
                layout=goal['puzzle_key'].rsplit('_',1)[0]; starts=initial[layout]
                full=summary['conditions']['full']; length=full['attempt_count']
                full_rows=[]
                for number,coords in enumerate(full['coords'],1):
                    a=recorded_attempt(r,'full',number,coords)
                    model_rows=score_attempt(summary=summary,unit_dir=out,condition='full',attempt=a)
                    truth=a['truth']['prediction_endpoints']['objects']
                    assert {v['object_id'] for v in model_rows}==set(truth)==set(starts)|{'PLACED'}
                    points={**starts,'PLACED':coords}
                    base=float(np.mean([pq_no_motion(points[k],endpoint(v)) for k,v in truth.items()]))
                    baseline_attempt=copy.deepcopy(a)
                    baseline_attempt['model_response']={'coordinate_predictions':[
                        {'id':k,'state':'in_scene','point':points[k],'exit_side':None,'orientation_deg':None} for k in truth]}
                    scored=score_attempt(summary=summary,unit_dir=out,condition='full',attempt=baseline_attempt)
                    close(base,np.mean([v['prediction_quality_sigma150']for v in scored]),'independent no-motion score')
                    observed=float(np.mean([v['prediction_quality_sigma150']for v in model_rows]))
                    row=dict(goal=gid,layout=layout,attempt=number,S=int(a['goal_succeeded']),
                             episode_n=length,weight=1/length,model=observed,no_motion=base,difference=observed-base)
                    attempts[m].append(row);full_rows.append(row);validated+=1
                assert len(full_rows)==length
                if gid in density_by:
                    rr=dict(goal=gid,layout=layout,p1=density_by[gid]['p1'],p8=density_by[gid]['p8'])
                    for arm in ['full','neither']:
                        branch=summary['conditions'][arm]
                        rr[arm]=int(branch['solved_by_budget']);rr[arm+'_minus_random']=rr[arm]-rr['p8']
                        rr[arm+'_first_success']=branch['first_success_attempt'] if branch['solved_by_budget'] else None
                    solves[m].append(rr)
                if gid not in novel_ids: continue
                side={x['source']:x['record'] for x in r['sidecar_records']}
                for phase,source in [('terminal_pre','pre_feedback/visual/terminal_probe'),
                                     ('terminal_post','full/terminal_prediction'),
                                     ('early_post','full/early_prediction')]:
                    q=side['transfer_sidecars/'+source+'/result.json']; probe=q['probe']
                    truth=probe['endpoints']; scored=q['score_rows']; points={**starts,'PLACED':probe['coords']}
                    assert {x['object_id']for x in scored}==set(truth)==set(points)
                    base=float(np.mean([pq_no_motion(points[k],v)for k,v in truth.items()]))
                    observed=float(np.mean([x['prediction_quality_sigma150']for x in scored]))
                    probes[m].append(dict(goal=gid,layout=layout,phase=phase,S_first=full_rows[0]['S'],
                                          model=observed,no_motion=base,difference=observed-base))
                pair=[x for x in probes[m][-3:]if x['phase'].startswith('terminal_')]
                close(pair[0]['no_motion'],pair[1]['no_motion'],'fixed terminal probe baseline')
        print('Baseline records:',m,shard['path'],flush=True) if shard is manifest['shards'][-1] else None

    random={}
    for label,rows in [('all_density_tasks',density),('main_density_tasks',[r for r in density if r['in_main']])]:
        random[label]={key:interval(rows,key)for key in ['p1','p8','p8_distinct']}
        random[label]['curves']=[interval([{**r,'value':uniform_search(r['p1'],b)}for r in rows],'value') for b in range(1,9)]
    known_sum=sum(density_by[g]['p8']for g in main if g in density_by)
    random['main_all_tasks_bounds']={'n':len(main),'known_density_tasks':1509,'unknown_current_density_tasks':132,
                                    'lower':known_sum/len(main),'upper':(known_sum+132)/len(main),
                                    'interpretation':'Bounds, not confidence intervals: allow canonical two-second random-search success to range from zero to one.'}
    random['model_comparison']={}
    random_csv=[]
    for m in MODELS[:3]:
        rows=solves[m]; assert len(rows)==1509
        random['model_comparison'][m]={key:interval(rows,key)for key in ['full','neither','full_minus_random','neither_minus_random']}
        for arm in ['full','neither']:
            random['model_comparison'][m][arm]['numerator']=sum(r[arm]for r in rows)
            curves=[]
            for budget in range(1,9):
                points=[{**r,'value':int(r[arm+'_first_success']is not None and r[arm+'_first_success']<=budget)}for r in rows]
                curves.append(interval(points,'value'))
            random['model_comparison'][m][arm+'_curve']=curves
            for key in [arm,arm+'_minus_random']:
                s=random['model_comparison'][m][key]
                random_csv.append(dict(model=m,metric=key,estimate=s['estimate'],ci95_low=s['ci95'][0],ci95_high=s['ci95'][1],tasks=s['tasks'],layouts=s['layouts']))

    no_motion={}; motion_csv=[]
    for m in MODELS:
        aa=attempts[m]; pp=probes[m]; no_motion[m]={}
        groups=[('chosen_initial',[r for r in aa if r['attempt']==1],None,20261114,10000),
                ('chosen_final',[r for r in aa if r['attempt']==r['episode_n']],None,20261114,10000),
                ('chosen_success',[r for r in aa if r['S']==1],'weight',6112701,2000),
                ('chosen_failure',[r for r in aa if r['S']==0],'weight',6112701,2000),
                ('novel_pre',[r for r in pp if r['phase']=='terminal_pre'],None,20261114,10000),
                ('novel_post',[r for r in pp if r['phase']=='terminal_post'],None,20261114,10000),
                ('early_novel_first_success',[r for r in pp if r['phase']=='early_post'and r['S_first']==1],None,6112701,2000),
                ('early_novel_first_failure',[r for r in pp if r['phase']=='early_post'and r['S_first']==0],None,6112701,2000)]
        for name,rr,weight,seed,draws in groups:
            z={key:interval(rr,key,weight,seed,draws)for key in ['model','no_motion','difference']}
            no_motion[m][name]=z
            for kind,s in z.items():
                motion_csv.append(dict(model=m,checkpoint=name,statistic=kind,estimate=s['estimate'],ci95_low=s['ci95'][0],ci95_high=s['ci95'][1],observations=s['n'],tasks=s['tasks'],layouts=s['layouts'],weighting='inverse episode length'if weight else'equal episode'))
            if name in ['chosen_success','chosen_failure']:
                reference=original[m]['figure4']['conditional_A']['1'if name=='chosen_success'else'0']
            elif name.startswith('early_novel_'):
                reference=original[m]['figure4']['conditional_C']['1'if name.endswith('_success')else'0']
            else:
                key={'chosen_initial':'chosen_pq_initial','chosen_final':'chosen_pq_final','novel_pre':'novel_pre','novel_post':'novel_post'}[name]
                reference=original[m]['interaction'][key]
            for field in ['estimate','ci95']: close(z['model'][field],reference[field],m+' '+name+' matches published '+field)

    report={'random_search':random,'no_motion':no_motion,
            'methods':{
                'random_search':'Average 1-(1-p_g)^8 across tasks, with p_g equal to successful/valid saved-grid placements. Each trial resets the same world. This is uniform grid search, not retry-only LMM behavior.',
                'random_distinct':'Sensitivity for eight distinct valid grid placements: 1-product_{j=0..7}(1-K_g/(N_g-j)).',
                'random_scope':'1560 noncanonical goals have compatible saved densities; 1509 are in the primary 1641-task set. Canonical132 has a saved three-second solution proxy, not a matched two-second density, and is excluded from the point baseline.',
                'random_uncertainty':'Intervals resample whole layouts, conditional on fixed saved-grid densities. They do not estimate discretization error or success-density uncertainty over continuous placements.',
                'no_motion':'Predict each scored dynamic object stays at its simulator initial center; predict PLACED stays at the supplied chosen/probe placement. Predict no exits. Static objects are excluded exactly as in model PQ. Uses exact scene geometry, without perceptual localization error.',
                'PQ':'Average exp(-squared_endpoint_error/(2*150^2)) over the same objects as the model score. Actual exits score zero for no-motion. Invalid/missing model forecasts retain the original zero credit.',
                'pairing':'Compare each model with no-motion on that model’s exact actions, probe placements, objects, checkpoints and task population. Differences are paired, not differences between independently sampled averages.',
                'intervals':'Pointwise whole-layout percentile bootstrap; retain all goals, directions and attempts. Conditional chosen groups use inverse episode-length weights matching Figure 4A. Other groups weight episodes equally.',
                'causal_scope':'Beating random search indicates useful action selection relative to this reference; it does not isolate feedback learning. Full-feedback versus retry remains the matched feedback-use comparison.',
                'qualitative_prediction':'The response schema requests terminal object states, endpoints, exit sides and optional orientation. Collision and movement events define goals/diagnostics; no separately scored prospective collision/movement prediction is part of these results.'},
            'verification':{'chosen_attempts_independently_rescored':validated,'saved_density_count_ratios_checked':len(density),
                            'published_model_means_and_intervals_reproduced':True,'fixed_terminal_probe_baseline_identical_pre_post':True},
            'inputs_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [repo/'benchmark/tasks.json',repo/'benchmark/asset_index.json',repo/'data/finalized/manifest.json']}}
    write(out/'summary.json',report)
    csv_out(out/'random_search.csv',random_csv)
    csv_out(out/'no_motion.csv',motion_csv)
    csv_out(out/'random_search_task_densities.csv',density)
    write_report(out,report)
    print('Baseline computation complete:',out,flush=True)
    return report

def write_report(out,report):
    names={'gpt5':'GPT-5','gemini31pro':'Gemini 3.1 Pro','qwen36plus':'Qwen 3.6 Plus','robotics_er2':'Gemini Robotics-ER 2'}
    def ci(s,percent=False):
        f=100 if percent else 1;d=2 if percent else 3
        return f"{f*s['estimate']:.{d}f} [{f*s['ci95'][0]:.{d}f}, {f*s['ci95'][1]:.{d}f}]"
    rs=report['random_search'];nm=report['no_motion'];base=rs['main_density_tasks']['p8']
    lines=['# Reference baselines','',
           'These references use existing model records, task densities and simulator initial states. No new model inference is required. All intervals are pointwise 95% whole-layout bootstrap intervals.','',
           '## Uniform random placement','',
           'For goal g, let p_g = K_g / N_g, the fraction of saved valid grid placements that solve it. The expected success probability within eight independent uniform placements is 1 - (1 - p_g)^8. We average this probability across tasks, rather than applying the formula to the average density.',
           '',
           f"On the **1,509 density-supported tasks within the 1,641-task main population**, expected random-search success is **{ci(base,True)}%**. Eight distinct placements without replacement give **{ci(rs['main_density_tasks']['p8_distinct'],True)}%**.",
           '', '| Method | Eight-attempt full-feedback success (%) | Retry-only success (%) |', '|---|---:|---:|',
           f"| Uniform random placement | {ci(base,True)} | same policy |"]
    for m in MODELS[:3]:
        v=rs['model_comparison'][m];lines.append(f"| {names[m]} | {ci(v['full'],True)} | {ci(v['neither'],True)} |")
    bounds=rs['main_all_tasks_bounds']
    lines += ['',
              f"Across all 1,560 noncanonical benchmark goals, the eight-attempt reference is {ci(rs['all_density_tasks']['p8'],True)}%. The 132 canonical goals have an older three-second solution-density proxy, while current success uses a two-second dwell. We therefore do not substitute that proxy into a supposedly matched full-population baseline.",
              f"Allowing the unknown canonical random-search success rates to range anywhere from zero to one bounds the 1,641-task mean at {100*bounds['lower']:.2f}%–{100*bounds['upper']:.2f}%. These are bounds, not confidence intervals.",
              'This baseline is uniform over the saved valid-placement grids, with each attempted action starting from the same initial world. The bootstrap describes variation across layouts conditional on the saved densities; it does not estimate continuous-placement discretization error. Model retry-only sampling is a separate, informed policy and outperforms this uniform reference. Beating random search does not by itself establish feedback learning; that question uses paired full-feedback versus retry controls.',
              '', '## No-motion prediction','',
              'Predict that each scored dynamic object stays at its initial simulator center and that the tool stays at the supplied placement. Predict no exits. Use the same objects, actions, novel probes and Gaussian PQ rule (sigma = 150 pixels) as each model. This reference uses exact initial geometry and has no perceptual-localization error. It is a heuristic, not a universal chance level.',
              '', '### Successful chosen actions','',
              'These are the Figure 4A successful-attempt populations, with inverse episode-length weights. Differences and intervals pair each model prediction with no-motion on the identical action.',
              '', '| Model | Successful tasks | Model PQ | No-motion PQ | Model minus no-motion |', '|---|---:|---:|---:|---:|']
    for m in MODELS:
        v=nm[m]['chosen_success'];lines.append(f"| {names[m]} | {v['model']['tasks']} | {ci(v['model'])} | {ci(v['no_motion'])} | {ci(v['difference'])} |")
    lines += ['', '### Terminal novel-action predictions','',
              '| Model | Tasks | Model PQ after interaction | No-motion PQ | Model minus no-motion |', '|---|---:|---:|---:|---:|']
    for m in MODELS:
        v=nm[m]['novel_post'];lines.append(f"| {names[m]} | {v['model']['tasks']} | {ci(v['model'])} | {ci(v['no_motion'])} | {ci(v['difference'])} |")
    lines += ['',
              'The three main models use the same 116 task IDs but can receive different probe placements, so their no-motion values differ. Robotics-ER 2 uses its separate 127-task probe population. For a fixed probe, no-motion is identical before and after feedback.',
              'Successful chosen-action predictions exceed no-motion in all four models. For terminal novel probes, Qwen has a positive paired interval; GPT-5, Gemini 3.1 Pro and Robotics-ER 2 have intervals spanning zero. This supports an above-reference chosen-prediction result and a model-dependent novel-prediction result, rather than equating a PQ near 0.5 with chance.',
              '', '## Scope of qualitative prediction','',
              'The released schema and primary PQ evaluate terminal object states, spatial endpoints and exit sides. Optional orientation and coarse terminal-location diagnostics exist. The movement and collision events used for goal generation and trajectory characterization are not separately scored prospective event predictions. Appropriate manuscript wording is “terminal object states and spatial endpoints,” without claiming a direct test of which objects will move or collide.',
              '', '## Files and reproduction','',
              '- `summary.json`: exact estimates, intervals, paired contrasts, curves and methods.',
              '- `random_search.csv`: matched model success and model-minus-random contrasts.',
              '- `random_search_task_densities.csv`: the exact task densities, supporting counts and membership.',
              '- `no_motion.csv`: initial/final chosen, successful/failed chosen, pre/post novel and one-rollout novel comparisons.',
              '- `random_search.tex` and `no_motion.tex`: standalone `booktabs` table fragments.',
              '', 'From the repository root:', '', '```bash',
              'python scripts/evaluation/baselines.py --results results --output outputs/baselines', '```',
              '', 'The main `python scripts/reproduce.py --output outputs/reproduction` command also regenerates these references.', '']
    (out/'README.md').write_text('\n'.join(lines))
    random_tex=['\\begin{tabular}{lrr}','\\toprule','Method & Full feedback (\\%) & Retry only (\\%) \\\\','\\midrule',f"Uniform random & {ci(base,True)} & {ci(base,True)} \\\\"]
    for m in MODELS[:3]:
        v=rs['model_comparison'][m];random_tex.append(f"{names[m]} & {ci(v['full'],True)} & {ci(v['neither'],True)} \\\\")
    random_tex+=['\\bottomrule','\\end{tabular}','% All rows use the same 1,509 density-supported main tasks; intervals resample layouts.']
    (out/'random_search.tex').write_text('\n'.join(random_tex)+'\n')
    tex=['\\begin{tabular}{llrrrr}','\\toprule','Prediction & Model & Tasks & Model PQ & No motion & Paired difference \\\\','\\midrule']
    for key,title in [('chosen_success','Successful chosen'),('novel_post','Terminal novel')]:
        for m in MODELS:
            v=nm[m][key];tex.append(f"{title} & {names[m]} & {v['model']['tasks']} & {ci(v['model'])} & {ci(v['no_motion'])} & {ci(v['difference'])} \\\\")
    tex+=['\\bottomrule','\\end{tabular}','% Every bracket is a pointwise 95-percent layout-bootstrap interval.']
    (out/'no_motion.tex').write_text('\n'.join(tex)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=Path,default=ROOT)
    p.add_argument('--results',type=Path,default=ROOT/'results');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze_baselines(a.repo_root.resolve(),a.results.resolve(),a.output.resolve())
