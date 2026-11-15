"""Independent high-draw feedback CIs, scoring sensitivity, and canonical correspondence."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,collections,csv,gzip,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--figure4-results',type=Path,required=True)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--output-dir',type=Path,required=True)
a=parser.parse_args();O=a.output_dir.resolve();R=a.repo_root.resolve()
if O.exists() and any(O.iterdir()):raise ValueError('Use a new/empty output directory')
(O/'results').mkdir(parents=True,exist_ok=True)
M=['gpt5','gemini31pro','qwen36plus'];D=json.loads((a.figure4_results/'reconstructed_models.json').read_text())
for units in D.values():
 for u in units.values():
  for arm in u['arms'].values():
   for x in arm['attempts']:x['success']=x['S']
pop=json.loads((R/'analysis/figure4/task_populations.json').read_text());core=pop['main1641'];canon=pop['canonical118']
import figure4_estimators as ae
env={k:getattr(ae,k) for k in ['regression','reg','partial']}
def boot(values,layouts,draws=100000):
 labels=sorted(set(layouts));s=np.array([sum(v for v,l in zip(values,layouts) if l==k) for k in labels]);n=np.array([layouts.count(k) for k in labels]);ix=np.random.default_rng(20261114).integers(len(labels),size=(draws,len(labels)));b=s[ix].sum(1)/n[ix].sum(1)
 return {'estimate':float(np.mean(values)),'ci95':np.quantile(b,[.025,.975]).tolist(),'n':len(values),'layouts':len(labels),'draws':draws,'seed':20261114}
out={'feedback_100k':{},'sigma_sensitivity':{},'canonical_correspondence':{}}
for m in M:
 v=D[m];fail=[g for g in core if not v[g]['arms']['full']['attempts'][0]['success']];early=[g for g in canon if not v[g]['arms']['full']['attempts'][0]['success']];b={}
 for label,ids,fn in [('subsequent_solve',fail,lambda u:u['arms']['full']['solved']-u['arms']['neither']['solved']),('next_chosen_pq',fail,lambda u:u['arms']['full']['attempts'][1]['pq150']-u['arms']['neither']['attempts'][1]['pq150']),('early_novel_pq',early,lambda u:u['transfer']['full']['early_post']['pq150']-u['transfer']['neither']['early_post']['pq150'])]:b[label]=boot([fn(v[g]) for g in ids],[v[g]['layout'] for g in ids])
 out['feedback_100k'][m]=b;out['sigma_sensitivity'][m]={}
 for s in [75,100,150]:
  rows=[{'goal':g,'layout':v[g]['layout'],'S':int(a['success']),'attempt':a['attempt'],'pq':a[f'pq{s}'],'episode_n':len(v[g]['arms']['full']['attempts'])} for g in core for a in v[g]['arms']['full']['attempts']]
  st={'figure4A':env['regression'](rows)}
  for name,ids,fn in [('next_chosen_after_failure',fail,lambda u:u['arms']['full']['attempts'][1][f'pq{s}']-u['arms']['neither']['attempts'][1][f'pq{s}']),('early_novel_after_failure',early,lambda u:u['transfer']['full']['early_post'][f'pq{s}']-u['transfer']['neither']['early_post'][f'pq{s}']),('terminal_novel_gain_beyond_retry',canon,lambda u:u['transfer']['full']['terminal_post'][f'pq{s}']-u['transfer']['full']['terminal_pre'][f'pq{s}']-u['transfer']['neither']['terminal_post'][f'pq{s}']+u['transfer']['neither']['terminal_pre'][f'pq{s}'])]:st[name]=boot([fn(v[g]) for g in ids],[v[g]['layout'] for g in ids],10000)
  out['sigma_sensitivity'][m][str(s)]=st
 print('feedback and scoring',m,flush=True)
(O/'results/additional_statistics.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
rows=[]
for m,scales in out['sigma_sensitivity'].items():
 for s,metrics in scales.items():
  for metric,x in metrics.items():rows.append({'model':m,'sigma_pixels':s,'metric':metric,'estimate':x['estimate'],'ci95_low':x['ci95'][0],'ci95_high':x['ci95'][1],'n':x['n'],'layouts':x['layouts']})
with (O/'results/scoring_sensitivity.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved independent statistics')
