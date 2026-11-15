#!/usr/bin/env python3
"""Reproduce the final binary-success Figure 4 model analyses from public shards.

No inference, manuscript files, or human records are read. The default compares
every coefficient and CI endpoint to the independently verified November 14
reference. Output must be new; original data and scientific outputs are read-only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import analyze_action_predictions as scorer
import figure4_estimators as est
from figure4_archive import archived_attempt, side_scores

MODELS=('gpt5','gemini31pro','qwen36plus')
ROOT=Path(__file__).resolve().parents[2]


def reconstruct(repo, output, model):
    snapshot='snapshot_seed2026_documented_gaps_2026-11-05'
    if model=='robotics_er2':snapshot='robotics_er2_canonical132_seed2026_2026-11-08'
    units={};hashes={}
    for path in sorted((repo/'model_outputs'/snapshot/'data'/model).glob('*.jsonl.gz')):
        hashes[str(path.relative_to(repo))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt') as f:
            for line in f:
                record=json.loads(line);summary=record['summary'];g=summary['goal'];gid=g['balanced_goal_id']
                assert gid not in units
                u=dict(goal=gid,layout=g['puzzle_key'].rsplit('_',1)[0],upward=int(g['condition']=='upward'),
                    seed=summary['seed'],arms={},transfer={},issues=[])
                assert u['seed']==2026
                for c,b in summary['conditions'].items():
                    attempts=[]
                    for n,coords in enumerate(b['coords'],1):
                        try:
                            a=archived_attempt(record,c,n,coords)
                            scores=scorer.score_attempt(summary=summary,unit_dir=output,condition=c,attempt=a)
                            assert scores
                            attempts.append(dict(attempt=n,coords=coords,S=int(a['goal_succeeded']),
                                **{f'pq{s}':float(np.mean([r[f'prediction_quality_sigma{s}'] for r in scores])) for s in (75,100,150)}))
                        except (AssertionError,KeyError) as exc:
                            u['issues'].append(dict(condition=c,attempt=n,error=type(exc).__name__))
                    u['arms'][c]=dict(solved=int(b['solved_by_budget']),attempts=attempts,count=b['attempt_count'])
                sides=[r['record'] for r in record['sidecar_records'] if r['source']=='transfer_sidecars/summary.json']
                assert len(sides)<=1
                if sides:
                    side=sides[0];pre=side['results']['pre_feedback'];sel=side['probe_selection']
                    u['early_probe_coords']=sel['early_probe']['coords']
                    for c,b in side['results']['conditions'].items():
                        rep='trace' if c=='trace_status' else 'visual'
                        u['transfer'][c]=dict(contaminated=bool(b['terminal_probe_contaminated']),
                            terminal_pre=side_scores(pre[f'{rep}_terminal_probe']),terminal_post=side_scores(b['terminal_prediction']),
                            early_pre=side_scores(pre[f'{rep}_early_probe']),early_post=side_scores(b['early_prediction']))
                units[gid]=u
    assert units,'No input shards found'
    return units,hashes


def observations(units, broad, canonical, condition='full', sigma=150):
    attempts=[];episodes=[];key=f'pq{sigma}'
    for gid in broad:
        u=units[gid];aa=u['arms'][condition]['attempts']
        assert len(aa)==u['arms'][condition]['count']
        assert bool(max(a['S'] for a in aa))==bool(u['arms'][condition]['solved'])
        for a in aa:
            attempts.append(dict(goal=gid,layout=u['layout'],attempt=a['attempt'],S=a['S'],pq=a[key],episode_n=len(aa)))
    for gid in canonical:
        u=units[gid];aa=u['arms'][condition]['attempts'];t=u['transfer'][condition]
        assert np.linalg.norm(np.array(u['early_probe_coords'])-aa[0]['coords'])>=40
        episodes.append(dict(goal=gid,layout=u['layout'],upward=u['upward'],S_first=aa[0]['S'],
            P_first=aa[0][key],P_final=aa[-1][key],N_pre=t['terminal_pre'][key],N=t['terminal_post'][key],
            exposure=len(aa),early_pre=t['early_pre'][key],early_post=t['early_post'][key],N_early=t['early_post'][key]))
    return attempts,episodes


def conditional_mean(rows,success):
    rr=[r for r in rows if r['S']==success];labs=sorted({r['layout']for r in rr})
    num=np.array([sum(r['pq']/r['episode_n']for r in rr if r['layout']==l)for l in labs])
    den=np.array([sum(1/r['episode_n']for r in rr if r['layout']==l)for l in labs])
    sample=np.random.default_rng(6112701).integers(len(labs),size=(2000,len(labs)))
    bs=num[sample].sum(axis=1)/den[sample].sum(axis=1)
    return dict(estimate=float(num.sum()/den.sum()),ci95=np.quantile(bs,[.025,.975]).tolist(),n=len(rr))


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    repo=a.repo_root.resolve();out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be new/empty')
    out.mkdir(parents=True,exist_ok=True)
    pop=json.loads((repo/'analysis/figure4/task_populations.json').read_text())
    expected=json.loads((repo/'analysis/figure4/verified_estimates_2026-11-14.json').read_text())
    result={};obs={'attempts':{},'episodes':{}};hashes={};units_all={}
    for model in MODELS:
        units,hh=reconstruct(repo,out,model);units_all[model]=units;hashes.update(hh)
        at,ep=observations(units,pop['main1641'],pop['canonical118'])
        obs['attempts'][model]=at;obs['episodes'][model]=ep
        r=dict(A=est.regression(at),B=est.partial(ep,['upward']),
            B_richer=est.partial(ep,['upward','P_first','N_pre','exposure']),
            C=est.reg(ep,'N_early','S_first',['early_pre','upward'],'layout'),
            conditional={'A':{str(s):conditional_mean(at,s) for s in (0,1)},
                'C':{str(s):est.simple([e for e in ep if e['S_first']==s],lambda rows:float(np.mean([x['early_post']for x in rows]))) for s in (0,1)}})
        for panel in ('A','B','B_richer','C'):
            x,y=r[panel],expected[model][panel]
            assert x['n']==y['n'] and x['layouts']==y['layouts'],(model,panel,'population mismatch')
            assert np.allclose([x['estimate']]+x['ci95'],[y['estimate']]+y['ci95'],rtol=0,atol=1e-10),(model,panel,x,y)
        result[model]=r
        print(model+': all 4 estimates and 8 CI bounds match audit',flush=True)
    common=sorted(set.intersection(*(set(v) for v in units_all.values())))
    eligible=[g for g in common if all(not units_all[m][g]['issues'] for m in MODELS)]
    assert len(common)==1651 and eligible==pop['main1641']
    for name,data in [('estimates',result),('model_observations',obs),('input_sha256',hashes),('reconstructed_models',units_all)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
    (out/'validation.json').write_text(json.dumps(dict(all_estimates_verified=True,absolute_tolerance=1e-10,
        common_completed=1651,eligible=1641,excluded=sorted(set(common)-set(eligible)),
        canonical=len(pop['canonical118']),seed=2026,bootstrap_draws=2000),indent=2)+'\n')


if __name__=='__main__':main()
