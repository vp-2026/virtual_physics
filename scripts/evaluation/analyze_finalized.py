#!/usr/bin/env python3
"""Reproduce finalized model estimates, intervals, populations and figure data."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from read_finalized import read_units,transfer_issues,MODELS
from estimation import bootstrap,label_seed,observations,conditional_mean
import figure4_estimators as est
ROOT=Path(__file__).resolve().parents[2]
MAIN=MODELS[:3]
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n')
def analyze(repo,out):
 if out.exists()and any(out.iterdir()):raise ValueError('Use a new/empty output directory')
 out.mkdir(parents=True,exist_ok=True);D,manifest=read_units(repo,out)
 old=json.loads((repo/'benchmark/evaluation_task_ids.json').read_text())
 broad=sorted(set.intersection(*(set(D[m])for m in MAIN)))
 main=[g for g in old['main_task_ids']if g in broad and all(not D[m][g]['issues'] for m in MAIN)]
 removed_transfer={g:{m:transfer_issues(D[m][g])if g in D[m]else ['missing_unit']for m in MAIN} for g in old['transfer_candidate_task_ids']}
 transfer=[g for g,reason in removed_transfer.items()if all(not x for x in reason.values())]
 removed_transfer={g:r for g,r in removed_transfer.items()if g not in transfer}
 pop={'main':main,'broad':broad,'matched_transfer':transfer,'canonical':sorted(g for g in broad if g.startswith('canonical_')),'removed_from_historical_main':[g for g in old['main_task_ids']if g not in main],'removed_from_historical_transfer':removed_transfer,'robotics_main':sorted(D['robotics_er2']),'robotics_transfer':sorted(g for g,u in D['robotics_er2'].items()if not u['issues']and not transfer_issues(u))}
 result={};table=[];sensitivity_table=[];obs={'attempts':{},'episodes':{}}
 def add(m,p,metric,s):
  table.append({'model':m,'population':p,'metric':metric,'estimate':s['estimate'],'ci95_low':s['ci95'][0],'ci95_high':s['ci95'][1],'n':s['n'],'n_tasks':s.get('n_tasks',s['n']),'layouts':s.get('layouts'),'numerator':s.get('numerator'),'bootstrap_draws':s.get('bootstrap_draws',s.get('valid_draws')),'bootstrap_seed':s.get('bootstrap_seed')})
 for m in MODELS:
  units=D[m];er=m=='robotics_er2';ids=pop['robotics_main']if er else main;tr=pop['robotics_transfer']if er else transfer;canonical=ids if er else pop['canonical'];r={'solve_rates':{},'interaction':{},'feedback':{},'canonical_controls':{},'figure4':{}}
  def mean(gg,fn,seed=20261114,draws=10000):return bootstrap([fn(units[g])for g in gg],[units[g]['layout']for g in gg],seed,draws)
  for population,gg in [('main',ids)]+([]if er else [('broad',broad)]):
   rates={}
   for c in ['full','neither','trace_status']:
    s=mean(gg,lambda u:u['arms'][c]['solved'],label_seed(f'fullbank-{m}-{c}-solve'if population=='broad'else f'audit-main-{m}-{c}-solve'));s['numerator']=sum(units[g]['arms'][c]['solved']for g in gg);rates[c]=s;add(m,population,c+'_solve_rate',s)
   for a,b in [('full','neither'),('trace_status','neither'),('trace_status','full')]:
    s=mean(gg,lambda u:u['arms'][a]['solved']-u['arms'][b]['solved']);rates[a+'_minus_'+b]=s;add(m,population,a+'_minus_'+b+'_solve',s)
   r['solve_rates'][population]=rates
  for name,fn in [('initial_success',lambda u:u['arms']['full']['attempts'][0]['S']),('chosen_pq_initial',lambda u:u['arms']['full']['attempts'][0]['pq150']),('chosen_pq_final',lambda u:u['arms']['full']['attempts'][-1]['pq150']),('chosen_pq_gain',lambda u:u['arms']['full']['attempts'][-1]['pq150']-u['arms']['full']['attempts'][0]['pq150'])]:
   s=mean(ids,fn,label_seed(f'audit-main-{m}-initial')if name=='initial_success'else 20261114)
   if name=='initial_success':s['numerator']=sum(fn(units[g])for g in ids)
   r['interaction'][name]=s;add(m,'main',name,s)
  for g in ids:
   for k in ['S','coords','pq150']:assert units[g]['arms']['full']['attempts'][0][k]==units[g]['arms']['neither']['attempts'][0][k]
  fail=[g for g in ids if not units[g]['arms']['full']['attempts'][0]['S']];early=[g for g in tr if not units[g]['arms']['full']['attempts'][0]['S']]
  for name,gg,fn in [('subsequent_solve',fail,lambda u:u['arms']['full']['solved']-u['arms']['neither']['solved']),('next_chosen_pq',fail,lambda u:u['arms']['full']['attempts'][1]['pq150']-u['arms']['neither']['attempts'][1]['pq150']),('early_novel_pq',early,lambda u:u['transfer']['full']['early_post']['pq150']-u['transfer']['neither']['early_post']['pq150'])]:
   s=mean(gg,fn,6112701,2000);r['feedback'][name]=s;add(m,'shared_first_failure_'+('transfer'if name=='early_novel_pq'else'main'),name,s)
  for name,fn in [('novel_pre',lambda u:u['transfer']['full']['terminal_pre']['pq150']),('novel_post',lambda u:u['transfer']['full']['terminal_post']['pq150']),('novel_gain',lambda u:u['transfer']['full']['terminal_post']['pq150']-u['transfer']['full']['terminal_pre']['pq150']),('novel_gain_beyond_retry',lambda u:u['transfer']['full']['terminal_post']['pq150']-u['transfer']['full']['terminal_pre']['pq150']-u['transfer']['neither']['terminal_post']['pq150']+u['transfer']['neither']['terminal_pre']['pq150'])]:
   s=mean(tr,fn);r['interaction'][name]=s;add(m,'transfer',name,s)
  conditions=['full','neither','trace_status']if er else ['full','neither','trace_status','status_only','frames_only']
  for c in conditions:
   s=mean(canonical,lambda u:u['arms'][c]['solved'],6112601,5000);s['numerator']=sum(units[g]['arms'][c]['solved']for g in canonical);r['canonical_controls'][c]=s;add(m,'canonical',c+'_solve_rate',s)
  for a,b in [('full','status_only'),('status_only','neither'),('frames_only','neither'),('full','frames_only')]:
   if a in conditions and b in conditions:
    s=mean(canonical,lambda u:u['arms'][a]['solved']-u['arms'][b]['solved'],6112601,5000);r['canonical_controls'][a+'_minus_'+b]=s;add(m,'canonical',a+'_minus_'+b,s)
  at,ep=observations(units,ids,tr);obs['attempts'][m]=at;obs['episodes'][m]=ep
  f={'A':est.regression(at)};f['A']['bootstrap_seed']=6112702
  if not er:
   f.update(B=est.partial(ep,['upward']),B_richer=est.partial(ep,['upward','P_first','N_pre','exposure']),C=est.reg(ep,'N_early','S_first',['early_pre','upward'],'layout'))
   for k in ['B','B_richer']:f[k]['bootstrap_seed']=6112703
   f['C']['bootstrap_seed']=6112702
  for k,s in f.items():add(m,'main'if k=='A'else'transfer','figure4_'+k,s)
  f['conditional_A']={str(s):conditional_mean(at,s)for s in (0,1)}
  f['conditional_C']={str(s):est.simple([e for e in ep if e['S_first']==s],lambda rr:float(np.mean([x['N_early']for x in rr])))for s in (0,1)}
  r['figure4']=f
  if not er:
   sensitivity={}
   for sigma in [75,100,150]:
    key=f'pq{sigma}';scaled_at,_=observations(units,ids,tr,sigma=sigma)
    sr={'figure4A':est.regression(scaled_at),'next_chosen_after_failure':mean(fail,lambda u:u['arms']['full']['attempts'][1][key]-u['arms']['neither']['attempts'][1][key]),'early_novel_after_failure':mean(early,lambda u:u['transfer']['full']['early_post'][key]-u['transfer']['neither']['early_post'][key]),'terminal_novel_gain_beyond_retry':mean(tr,lambda u:u['transfer']['full']['terminal_post'][key]-u['transfer']['full']['terminal_pre'][key]-u['transfer']['neither']['terminal_post'][key]+u['transfer']['neither']['terminal_pre'][key])}
    sr['figure4A']['bootstrap_seed']=6112702;sensitivity[str(sigma)]=sr
    for metric,s in sr.items():sensitivity_table.append({'model':m,'sigma_pixels':sigma,'metric':metric,'estimate':s['estimate'],'ci95_low':s['ci95'][0],'ci95_high':s['ci95'][1],'tasks':s.get('n_tasks',s['n']),'layouts':s['layouts'],'bootstrap_draws':s['bootstrap_draws'],'bootstrap_seed':s['bootstrap_seed']})
   r['pq_scale_sensitivity']=sensitivity
  result[m]=r;print(m,'main',len(ids),'transfer',len(tr),'complete',flush=True)
 write(out/'summary.json',result);write(out/'populations.json',pop);write(out/'observations.json',obs)
 with (out/'estimates.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
 with (out/'pq_scale_sensitivity.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(sensitivity_table[0]),lineterminator='\n');w.writeheader();w.writerows(sensitivity_table)
 write(out/'method.json',{'dataset_manifest_sha256':hashlib.sha256((repo/'data/finalized/manifest.json').read_bytes()).hexdigest(),'PQ':'Mean object-level exp(-endpoint_error_squared/(2*150^2)); correct exit state/side=1, mismatch=0. Missing/invalid object forecasts receive zero under the frozen scorer; explicit provider-error probes are excluded as missing requests. Not percentage accuracy.','intervals':'Whole-layout cluster percentile bootstrap; both directions and all goals resampled together. Seeds/draw counts recorded per estimate.','main':'Frozen historical 1641-task membership intersected with complete, prediction-reconciled replacement records across all three main models.','broad':'Common completed model tasks, historically 1651.','transfer':'Frozen historical 118-task membership; require complete full/retry/JSON probes. Terminal placements must be at least 40px from every executed action, and early placements at least 40px from the first action in each arm.','robotics':'Separate canonical 132-task extension; transfer uses its own valid untried-probe population.','feedback':'Full minus retry paired on the same failed first action. JSON has its own initial action and is a representation control.','stopping':'Legacy main-model branches absorb immediate shared success; after first failure, status-hidden branches continue to eight. ER2 status-hidden branches always continue to eight.','human_data':'No participant data or idealized human estimates are used.'})
 return result,pop
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=Path,default=ROOT);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.repo_root.resolve(),a.output_dir.resolve())
