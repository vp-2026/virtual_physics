"""Reconcile populations and estimands using freshly rescored model archives."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,collections,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
import argparse
from analyze_figure4_correspondence import reconstruct
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--figure4-results',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args();R=args.repo_root.resolve();O=args.output_dir.resolve()
if O.exists() and any(O.iterdir()):raise ValueError('Use a new/empty output directory')
(O/'results').mkdir(parents=True,exist_ok=True)
M=['gpt5','gemini31pro','qwen36plus']
D=json.loads((args.figure4_results/'reconstructed_models.json').read_text())
D['robotics'],erhash=reconstruct(R,O,'robotics_er2')
for units in D.values():
 for gid,u in units.items():
  u['source']='canonical_world_gcond' if gid.startswith('canonical_') else 'noncanonical'
  for arm in u['arms'].values():
   for a in arm['attempts']:a['success']=a['S']
recon={'eligible_goal_ids':json.loads((R/'analysis/figure4/task_populations.json').read_text())['main1641']}
common=sorted(set.intersection(*(set(D[m]) for m in M)));core=recon['eligible_goal_ids']
def seed(label):return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4],'big')
def estimate(values,layouts,label,draws=10000,rngseed=None):
 group=collections.defaultdict(list)
 for v,l in zip(values,layouts):group[l].append(float(v))
 keys=sorted(group);sums=np.array([sum(group[k]) for k in keys]);ns=np.array([len(group[k]) for k in keys]);rng=np.random.default_rng(seed(label) if rngseed is None else rngseed)
 ix=rng.integers(len(keys),size=(draws,len(keys)));boot=sums[ix].sum(1)/ns[ix].sum(1)
 return {'estimate':float(sums.sum()/ns.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n':int(ns.sum()),'layouts':len(keys),'draws':draws,'bootstrap_seed':seed(label) if rngseed is None else rngseed}
# Frozen function-only estimators and task-only memberships live in this audit.
from figure4_estimators import regression as fe
POP=json.loads((R/'analysis/figure4/task_populations.json').read_text());report={'main':{},'broad':{},'canonical_decomposition':{},'stopping':{},'figure4A':{},'scale_sensitivity':{}};csvrows=[]
def add(model,panel,metric,pop,stat,numerator=None,reported=None,edit='Retain estimate; state the exact population and seed.'):
 csvrows.append({'model':model,'panel':panel,'metric':metric,'reported_value':reported,'recomputed_value':stat['estimate'],'numerator':numerator,'denominator':stat['n'],'layout_clusters':stat['layouts'],'ci95_low':stat['ci95'][0],'ci95_high':stat['ci95'][1],'seed_population':'2026 only','estimator':metric,'population':pop,'bootstrap_draws':stat['draws'],'bootstrap_seed':stat['bootstrap_seed'],'source_path':'public shards + Figure4 reconstructed_models.json','required_edit':edit})
for m in M:
 v=D[m];report['main'][m]={};report['broad'][m]={}
 for label,ids in [('main',core),('broad',common)]:
  ls=[v[g]['layout'] for g in ids]
  for c in ['full','neither','trace_status']:
   vals=[v[g]['arms'][c]['solved'] for g in ids];stat=estimate(vals,ls,f'fullbank-{m}-{c}-solve' if label=='broad' else f'audit-main-{m}-{c}-solve')
   report[label][m][c]=stat;add(m,'Figure3A' if label=='main' else 'Broad feedback table',c+' episode solve probability',f'{len(ids)} common '+('prediction-reconciled' if label=='main' else 'completed')+' tasks',stat,sum(vals))
  vals=[v[g]['arms']['full']['solved']-v[g]['arms']['neither']['solved'] for g in ids]
  stat=estimate(vals,ls,f'fullbank-{m}-net-solve' if label=='broad' else f'audit-main-{m}-net-solve');report[label][m]['full_minus_retry']=stat
  add(m,'Broad feedback table' if label=='broad' else 'All-task sensitivity','Full minus retry, paired episode solve probability',str(len(ids))+' common tasks',stat,sum(vals))
 ls=[v[g]['layout'] for g in core];first=[v[g]['arms']['full']['attempts'][0]['success'] for g in core]
 initial=estimate(first,ls,f'audit-main-{m}-initial');report['main'][m]['initial']=initial;add(m,'Figure3A','First-attempt success','1641 prediction-reconciled common tasks',initial,sum(first))
 fail=[g for g in core if not v[g]['arms']['full']['attempts'][0]['success']]
 for g in core:
  for k in ['coords','success','pq150']:assert v[g]['arms']['full']['attempts'][0][k]==v[g]['arms']['neither']['attempts'][0][k]
 vals=[v[g]['arms']['full']['solved']-v[g]['arms']['neither']['solved'] for g in fail]
 stat=estimate(vals,[v[g]['layout'] for g in fail],f'audit-firstfailure-{m}',2000,rngseed=6112701);report['main'][m]['conditional_solve_benefit']=stat;add(m,'Figure3B','Full minus retry, subsequent episode success conditional on shared first failure','Model-specific initial-failure subset of 1641 tasks',stat,sum(vals))
 vals=[v[g]['arms']['full']['attempts'][1]['pq150']-v[g]['arms']['neither']['attempts'][1]['pq150'] for g in fail]
 stat=estimate(vals,[v[g]['layout'] for g in fail],f'audit-nextpq-{m}',2000,rngseed=6112701);report['main'][m]['next_pq_benefit']=stat;add(m,'Figure3B','Chosen PQ at attempt 2: full minus retry','Same shared-first-failure subset',stat)
 canon=[g for g in v if v[g]['source']=='canonical_world_gcond'];ls=[v[g]['layout'] for g in canon]
 report['canonical_decomposition'][m]={}
 for c in ['full','status_only','frames_only','neither','trace_status']:
  vals=[v[g]['arms'][c]['solved'] for g in canon];stat=estimate(vals,ls,f'canonical-{m}-{c}',5000,rngseed=6112601);report['canonical_decomposition'][m][c]=stat;add(m,'Canonical feedback decomposition',c+' solve by budget','132 canonical tasks',stat,sum(vals))
 for c1,c2 in [('full','status_only'),('status_only','neither'),('frames_only','neither'),('full','frames_only')]:
  vals=[v[g]['arms'][c1]['solved']-v[g]['arms'][c2]['solved'] for g in canon];stat=estimate(vals,ls,'canonical',5000,rngseed=6112601);report['canonical_decomposition'][m][c1+'_minus_'+c2]=stat;add(m,'Canonical feedback decomposition',c1+' minus '+c2,'132 canonical tasks',stat,sum(vals))
 report['stopping'][m]={}
 for c in ['full','neither','frames_only','status_only','trace_status']:
  units=[u for u in v.values() if c in u['arms']]
  report['stopping'][m][c]={'episodes':len(units),'shorter_than_eight':sum(u['arms'][c]['count']<8 for u in units),'shorter_with_failed_first':sum(u['arms'][c]['count']<8 and not u['arms'][c]['attempts'][0]['success'] for u in units),'shorter_and_unsolved':sum(u['arms'][c]['count']<8 and not u['arms'][c]['solved'] for u in units)}
 rows=[{'goal':g,'layout':v[g]['layout'],'S':int(a['success']),'attempt':a['attempt'],'pq':a['pq150'],'episode_n':len(v[g]['arms']['full']['attempts'])} for g in core for a in v[g]['arms']['full']['attempts']]
 report['figure4A'][m]=fe(rows)
 report['scale_sensitivity'][m]={}
 for sigma in [75,100,150]:
  values=[v[g]['arms']['full']['attempts'][-1][f'pq{sigma}']-v[g]['arms']['full']['attempts'][0][f'pq{sigma}']-(v[g]['arms']['neither']['attempts'][-1][f'pq{sigma}']-v[g]['arms']['neither']['attempts'][0][f'pq{sigma}']) for g in core]
  stat=estimate(values,[v[g]['layout'] for g in core],f'audit-sigma-{m}-{sigma}');report['scale_sensitivity'][m][str(sigma)]={'chosen_gain_beyond_retry':stat}
 print('main/broad/control reconstructed',m,flush=True)
# Exact 118-task membership is taken from the frozen model analysis, not regenerated from human records.
membership=POP['canonical118']
assert len(membership)==118
report['matched118']={}
for m in M:
 v=D[m];ls=[v[g]['layout'] for g in membership];dd={}
 for k in ['terminal_pre','terminal_post']:
  dd[k]=estimate([v[g]['transfer']['full'][k]['pq150'] for g in membership],ls,f'{m}-118-{k}')
 dd['gain']=estimate([v[g]['transfer']['full']['terminal_post']['pq150']-v[g]['transfer']['full']['terminal_pre']['pq150'] for g in membership],ls,f'{m}-118-gain')
 fail=[g for g in membership if not v[g]['arms']['full']['attempts'][0]['success']]
 dd['early_benefit_first_failures']=estimate([v[g]['transfer']['full']['early_post']['pq150']-v[g]['transfer']['neither']['early_post']['pq150'] for g in fail],[v[g]['layout'] for g in fail],'early',2000,rngseed=6112701)
 add(m,'Figure3B','Novel PQ after exactly one rollout: full minus retry','Initial failures within frozen matched 118 tasks',dd['early_benefit_first_failures'])
 report['matched118'][m]=dd
# ER2: separately retain its own clean population and the historical matched 125 selection.
er=D['robotics'];allids=sorted(er);clean=[g for g in allids if all(not er[g]['transfer'][c]['contaminated'] for c in ['full','neither','trace_status'])]
matched125=POP['er2_historical125']
erreport={'model':'gemini-robotics-er-2-preview','episodes':len(er),'complete_probes':sum(bool(u['transfer']) for u in er.values()),'solved':sum(u['arms']['full']['solved'] for u in er.values()),'own_clean_n':len(clean),'historical_matched_n':len(matched125),'historical_selection_note':'The 125-task population is an intersection with the older specialist-model clean set; it is not ER2-only eligibility. No older-specialist performance is reported.','populations':{},'figure4A':{}}
for label,ids in [('own_clean127',clean),('historical_matched125',matched125)]:
 ls=[er[g]['layout'] for g in ids];d={}
 for key,name in [('terminal_pre','pre'),('terminal_post','post')]:d[name]=estimate([er[g]['transfer']['full'][key]['pq150'] for g in ids],ls,'robotics-robotics-'+name)
 d['net']=estimate([(er[g]['transfer']['full']['terminal_post']['pq150']-er[g]['transfer']['full']['terminal_pre']['pq150'])-(er[g]['transfer']['neither']['terminal_post']['pq150']-er[g]['transfer']['neither']['terminal_pre']['pq150']) for g in ids],ls,'robotics-robotics-net')
 d['raw_gain']=estimate([er[g]['transfer']['full']['terminal_post']['pq150']-er[g]['transfer']['full']['terminal_pre']['pq150'] for g in ids],ls,'robotics-raw')
 erreport['populations'][label]=d
for m in ['robotics','gemini31pro']:
 ids=[g for g in allids if g in D[m] and not D[m][g]['issues']]
 for c in ['full','trace_status']:
  rows=[{'goal':g,'layout':D[m][g]['layout'],'S':int(a['success']),'attempt':a['attempt'],'pq':a['pq150'],'episode_n':len(D[m][g]['arms'][c]['attempts'])} for g in ids for a in D[m][g]['arms'][c]['attempts']]
  erreport['figure4A'][m+'_'+c]=fe(rows)
(O/'results/solve_rate_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
(O/'results/er2_analysis.json').write_text(json.dumps(erreport,indent=2)+'\n')
with (O/'SOLVE_RATE_RECONCILIATION.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
print(json.dumps({'primary':{m:{k:report['main'][m][k] for k in ['initial','full','conditional_solve_benefit']} for m in M},'er2':erreport},indent=2))

(O/'results/er2_input_sha256.json').write_text(json.dumps(erhash,indent=2)+'\n')
