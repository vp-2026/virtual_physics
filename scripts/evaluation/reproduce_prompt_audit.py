#!/usr/bin/env python3
"""Recompute historical paired control estimates from released model-only scores.

This is a derived-score reproduction, not a new raw-response re-score of all
historical GPT/Gemini trials. See the source-directory README for provenance.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new/empty output directory')
    src=ROOT/'model_outputs/historical_prompt_control_audit_2026-11-14'
    rows=list(csv.DictReader((src/'paired_scores.csv').open()))
    for r in rows:
        for k in ('full_success','compact_success','full_pq','compact_pq'):r[k]=float(r[k])
    stats=[]
    for m in ('gpt5','gemini31pro','qwen36plus'):
        rr=[r for r in rows if r['model']==m];labs=sorted({r['layout'] for r in rr})
        for metric in ('success','pq'):
            num=np.array([sum(r['full_'+metric]-r['compact_'+metric] for r in rr if r['layout']==l) for l in labs])
            den=np.array([sum(r['layout']==l for r in rr) for l in labs])
            ix=np.random.default_rng(20261114).integers(len(labs),size=(10000,len(labs)))
            boot=num[ix].sum(1)/den[ix].sum(1)
            stats.append(dict(model=m,metric=metric,n=len(rr),layouts=len(labs),full=float(np.mean([r['full_'+metric] for r in rr])),compact=float(np.mean([r['compact_'+metric] for r in rr])),difference=float(num.sum()/den.sum()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    stale=json.loads((src/'qwen_shared_vs_accepted.json').read_text())['rows']
    mismatch=sum(r['coordinate_mismatch'] for r in stale)
    assert mismatch==36 and len(stale)==132
    compact_old=float(np.mean([r['shared_pq'] for r in stale]))
    compact_new=float(np.mean([r['accepted_branch_pq'] for r in stale]))
    q=next(r for r in stats if r['model']=='qwen36plus' and r['metric']=='pq')
    assert abs(q['compact']-compact_new)<1e-12
    assert abs(q['full']-compact_old-(-.013016295445173411))<1e-12
    out.mkdir(parents=True,exist_ok=True)
    (out/'prompt_control.json').write_text(json.dumps(dict(summary=stats,bootstrap_draws=10000,bootstrap_seed=20261114,cluster='layout',qwen_stale_compact_records=mismatch,old_shared_state_difference=q['full']-compact_old),indent=2)+'\n')
    print('Reproduced six paired estimates/CIs and the stale-join discrepancy')

if __name__=='__main__':main()
