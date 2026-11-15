#!/usr/bin/env python3
"""Independent endpoint scoring and estimator checks; no collection-code imports."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,collections,gzip,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2]

def close(a,b,label):
 assert np.allclose(a,b,rtol=0,atol=1e-9,equal_nan=False),(label,a,b)

def endpoint(x,y):
 inside=0<=x<=599 and 0<=y<=599
 side=max([(max(0,-x),'left'),(max(0,x-599),'right'),(max(0,-y),'bottom'),(max(0,y-599),'top')],key=lambda z:z[0])[1]
 return {'state':'in_scene'if inside else'exited','point':[x,y],'exit_side':None if inside else side}

def quality(pred,truth):
 if pred.get('state')!=truth['state']:return 0.
 if truth['state']=='exited':return float(pred.get('exit_side')==truth.get('exit_side'))
 point=pred.get('point')
 if not isinstance(point,list)or len(point)!=2 or not all(isinstance(v,(int,float))for v in point):return 0.
 return math.exp(-sum((a-b)**2 for a,b in zip(point,truth['point']))/(2*150**2))

def cluster_mean(values,labels,spec):
 groups=collections.defaultdict(list)
 for x,l in zip(values,labels):groups[l].append(x)
 names=sorted(groups);totals=np.array([sum(groups[l])for l in names]);sizes=np.array([len(groups[l])for l in names]);rng=np.random.default_rng(spec['bootstrap_seed']);draw=rng.integers(len(names),size=(spec['bootstrap_draws'],len(names)))
 close(np.mean(values),spec['estimate'],'mean estimate');close(np.quantile(totals[draw].sum(1)/sizes[draw].sum(1),[.025,.975]),spec['ci95'],'mean interval')

def partial_independent(rows,covariates):
 names=['P_final','N']+covariates;z=np.array([[r[k]for k in names]for r in rows]);layouts=np.array([r['layout']for r in rows]);groups=[np.where(layouts==g)[0]for g in sorted(set(layouts))]
 def fit(values,labels):
  ranks=rankdata(values,axis=0);levels=sorted(set(labels));dummies=np.array([[int(x==l)for l in levels]for x in labels],float);X=np.column_stack([dummies,ranks[:,2:]])
  residual=ranks[:,:2]-X@np.linalg.lstsq(X,ranks[:,:2],rcond=None)[0]
  return np.corrcoef(residual.T)[0,1]
 point=fit(z,layouts);rng=np.random.default_rng(6112703);samples=[]
 for _ in range(2000):
  chosen=rng.integers(len(groups),size=len(groups));ix=np.concatenate([groups[k]for k in chosen]);labels=np.concatenate([np.full(len(groups[k]),j)for j,k in enumerate(chosen)])
  samples.append(fit(z[ix],labels))
 return point,np.quantile(np.array(samples)[np.isfinite(samples)],[.025,.975])

def regression_independent(rows,panel):
 # Frisch-Waugh-Lovell: residualize the target against nuisance variables.
 if panel=='A':
  x=np.array([[r['S']]+[float(r['attempt']==k)for k in range(2,9)]for r in rows]);y=np.array([r['pq']for r in rows]);group=[r['goal']for r in rows];w=np.array([1/r['episode_n']for r in rows])
 else:
  x=np.array([[r[k]for k in ['S_first','early_pre','upward']]for r in rows]);y=np.array([r['N_early']for r in rows]);group=[r['layout']for r in rows];w=np.ones(len(rows))
 for g in sorted(set(group)):
  ix=np.array([a==g for a in group]);x[ix]-=x[ix].mean(0);y[ix]-=y[ix].mean()
 x=x[:,x.std(0)>1e-11];x*=np.sqrt(w[:,None]);y*=np.sqrt(w)
 layouts=np.array([r['layout']for r in rows]);names=sorted(set(layouts));xx=[];xy=[]
 for l in names:
  ix=layouts==l;xx.append(x[ix].T@x[ix]);xy.append(x[ix].T@y[ix])
 xx=np.array(xx);xy=np.array(xy)
 def coefficient(XX,XY):
  if len(XY)==1:return XY[0]/XX[0,0]
  residual_cross=XY[0]-XX[0,1:]@np.linalg.solve(XX[1:,1:],XY[1:])
  residual_square=XX[0,0]-XX[0,1:]@np.linalg.solve(XX[1:,1:],XX[1:,0])
  return residual_cross/residual_square
 point=coefficient(xx.sum(0),xy.sum(0));rng=np.random.default_rng(6112702);samples=[]
 for _ in range(2000):
  draw=rng.integers(len(names),size=len(names));a=xx[draw].sum(0);b=xy[draw].sum(0)
  if np.linalg.matrix_rank(a)==len(b):samples.append(coefficient(a,b))
 return point,np.quantile(samples,[.025,.975])

def verify(repo,results):
 S=json.loads((results/'summary.json').read_text());P=json.loads((results/'populations.json').read_text());O=json.loads((results/'observations.json').read_text());manifest=json.loads((repo/'data/finalized/manifest.json').read_text());models=['gpt5','gemini31pro','qwen36plus','robotics_er2'];units={m:{}for m in models};checked_attempts=0;checked_probes=0
 obs={(m,r['goal'],r['attempt']):r for m,rows in O['attempts'].items()for r in rows}
 for shard in manifest['shards']:
  model=shard['model'];main=set(P['robotics_main']if model=='robotics_er2'else P['main'])
  with gzip.open(repo/'data/finalized'/shard['path'],'rt')as stream:
   for line in stream:
    record=json.loads(line);s=record['summary'];g=s['goal'];gid=g['balanced_goal_id'];u={'layout':g['puzzle_key'].rsplit('_',1)[0],'arms':{},'probe_scores':{},'transfer_eligible':False}
    sides=[x['record']for x in record['sidecar_records']if x['source']=='transfer_sidecars/summary.json']
    if sides:
     side=sides[0];selection=side['probe_selection'];eligible=True
     bad_queries={x['source']for x in record['sidecar_records']if x['source'].endswith('/result.json')and not x['record'].get('schema_valid')and(x['record'].get('last_provider_metadata')or{}).get('finish_reason')=='error'}
     for arm in ['full','neither','trace_status']:
      b=side['results']['conditions'].get(arm);rep='trace'if arm=='trace_status'else'visual'
      if b is None:eligible=False;continue
      queries=[f'pre_feedback/{rep}/early_probe',f'pre_feedback/{rep}/terminal_probe',f'{arm}/early_prediction',f'{arm}/terminal_prediction']
      eligible &= not any('transfer_sidecars/'+q+'/result.json'in bad_queries for q in queries)
      eligible &= not b['terminal_probe_contaminated']
      eligible &= all(math.dist(selection['terminal_probe']['coords'],xy)>=40 for xy in s['conditions'][arm]['coords'])
      eligible &= math.dist(selection['early_probe']['coords'],s['conditions'][arm]['coords'][0])>=40
     u['transfer_eligible']=bool(eligible)
    for arm,summary in s['conditions'].items():
     u['arms'][arm]=bool(summary['solved_by_budget'])
     if gid not in main:continue
     successes=[];scores=[]
     for t,coords in enumerate(summary['coords'],1):
      prefix=('branches/trace_status/independent_attempt_1/'if arm=='trace_status'else'shared_attempt_1/')if t==1 else'branches/'+arm+'/'
      aa=[a for a in record['attempt_records']if a['source'].startswith(prefix)and a['attempt']==t and a['coords']==coords and not a.get('obstruction_detected')];assert len(aa)==1
      a=aa[0];label='shared_attempt_1'if t==1 else f'{arm}_attempt_{t}'
      calls=[c for c in record['provider_calls']if c['source'].startswith(prefix)and c['call_label']==label and ((c.get('parsed_action_payload')or{}).get('parsed')or{}).get('point')==coords];assert calls
      pred={x['id']:x for x in (calls[-1]['parsed_action_payload'].get('coordinate_predictions')or[])if isinstance(x,dict)and isinstance(x.get('id'),str)};truth=a['truth']['prediction_endpoints']['objects'];pq=np.mean([quality(pred.get(k,{}),endpoint(v['final_x'],v['final_y']))for k,v in truth.items()]);successes.append(bool(a['goal_succeeded']));scores.append(pq);checked_attempts+=1
      if arm=='full':close([pq,successes[-1]],[obs[model,gid,t]['pq'],obs[model,gid,t]['S']],'raw chosen prediction')
     assert any(successes)==u['arms'][arm];u[arm]={'S':successes,'PQ':scores}
    for item in record['sidecar_records']:
     if not item['source'].endswith('/result.json'):continue
     r=item['record'];pred={x['id']:x for x in ((r.get('parsed_payload')or{}).get('coordinate_predictions')or[])if isinstance(x,dict)and isinstance(x.get('id'),str)};truth=r['probe']['endpoints']
     for row in r.get('score_rows',[]):close(quality(pred.get(row['object_id'],{}),truth[row['object_id']]),row['prediction_quality_sigma150'],'raw novel prediction');checked_probes+=1
     if r.get('score_rows'):u['probe_scores'][item['source']]=float(np.mean([quality(pred.get(row['object_id'],{}),truth[row['object_id']])for row in r['score_rows']]))
    units[model][gid]=u
 historical=json.loads((repo/'archive/pre_correction_2026-11-14/analysis/figure4/task_populations.json').read_text())
 expected_transfer=sorted(g for g in historical['canonical118']if all(g in units[m]and units[m][g]['transfer_eligible']for m in models[:3]))
 assert expected_transfer==sorted(P['matched_transfer']),'Independent matched-probe eligibility mismatch'
 assert sorted(g for g,u in units['robotics_er2'].items()if u['transfer_eligible'])==sorted(P['robotics_transfer'])
 for m in models:
  ids=P['robotics_main']if m=='robotics_er2'else P['main'];U=units[m]
  for name,gg in [('main',ids)]+([]if m=='robotics_er2'else[('broad',P['broad'])]):
   for arm in ['full','neither','trace_status']:
    values=[U[g]['arms'][arm]for g in gg];spec=S[m]['solve_rates'][name][arm];assert sum(values)==spec['numerator']and len(values)==spec['n'];cluster_mean(values,[U[g]['layout']for g in gg],spec)
   for a,b in [('full','neither'),('trace_status','neither'),('trace_status','full')]:cluster_mean([int(U[g]['arms'][a])-int(U[g]['arms'][b])for g in gg],[U[g]['layout']for g in gg],S[m]['solve_rates'][name][a+'_minus_'+b])
  initial=[U[g]['full']['S'][0]for g in ids];cluster_mean(initial,[U[g]['layout']for g in ids],S[m]['interaction']['initial_success'])
  for k,fn in [('chosen_pq_initial',lambda u:u['full']['PQ'][0]),('chosen_pq_final',lambda u:u['full']['PQ'][-1]),('chosen_pq_gain',lambda u:u['full']['PQ'][-1]-u['full']['PQ'][0])]:cluster_mean([fn(U[g])for g in ids],[U[g]['layout']for g in ids],S[m]['interaction'][k])
  failed=[g for g in ids if not U[g]['full']['S'][0]]
  for k,fn in [('subsequent_solve',lambda u:int(u['arms']['full'])-int(u['arms']['neither'])),('next_chosen_pq',lambda u:u['full']['PQ'][1]-u['neither']['PQ'][1])]:cluster_mean([fn(U[g])for g in failed],[U[g]['layout']for g in failed],S[m]['feedback'][k])
  tr=P['robotics_transfer']if m=='robotics_er2'else P['matched_transfer']
  def probe(u,path):return u['probe_scores']['transfer_sidecars/'+path+'/result.json']
  for k,fn in [('novel_pre',lambda u:probe(u,'pre_feedback/visual/terminal_probe')),('novel_post',lambda u:probe(u,'full/terminal_prediction')),('novel_gain',lambda u:probe(u,'full/terminal_prediction')-probe(u,'pre_feedback/visual/terminal_probe')),('novel_gain_beyond_retry',lambda u:probe(u,'full/terminal_prediction')-probe(u,'neither/terminal_prediction'))]:cluster_mean([fn(U[g])for g in tr],[U[g]['layout']for g in tr],S[m]['interaction'][k])
  early=[g for g in tr if not U[g]['full']['S'][0]];cluster_mean([probe(U[g],'full/early_prediction')-probe(U[g],'neither/early_prediction')for g in early],[U[g]['layout']for g in early],S[m]['feedback']['early_novel_pq'])
  canon=ids if m=='robotics_er2'else P['canonical']
  for k,spec in S[m]['canonical_controls'].items():
   if '_minus_'in k:
    a,b=k.split('_minus_');values=[int(U[g]['arms'][a])-int(U[g]['arms'][b])for g in canon]
   else:values=[int(U[g]['arms'][k])for g in canon]
   cluster_mean(values,[U[g]['layout']for g in canon],spec)
  for panel,rows,value_key,success_key in [('A',O['attempts'][m],'pq','S'),('C',O['episodes'][m],'N_early','S_first')]:
   for success in [0,1]:
    selected=[r for r in rows if r[success_key]==success];layouts=sorted({r['layout']for r in selected});num=[];den=[]
    for l in layouts:
     rr=[r for r in selected if r['layout']==l];weights=[1/r['episode_n']if panel=='A'else 1 for r in rr];num.append(sum(r[value_key]*w for r,w in zip(rr,weights)));den.append(sum(weights))
    num=np.array(num);den=np.array(den);ix=np.random.default_rng(6112701).integers(len(layouts),size=(2000,len(layouts)));stat=S[m]['figure4']['conditional_'+panel][str(success)];close(num.sum()/den.sum(),stat['estimate'],'conditional mean');close(np.quantile(num[ix].sum(1)/den[ix].sum(1),[.025,.975]),stat['ci95'],'conditional interval')
  for panel,rows in [('A',O['attempts'][m])]+([]if m=='robotics_er2'else[('C',O['episodes'][m])]):
   point,interval=regression_independent(rows,panel);close(point,S[m]['figure4'][panel]['estimate'],panel+' independent point');close(interval,S[m]['figure4'][panel]['ci95'],panel+' independent interval')
  if m!='robotics_er2':
   for k,covs in [('B',['upward']),('B_richer',['upward','P_first','N_pre','exposure'])]:
    point,interval=partial_independent(O['episodes'][m],covs);close(point,S[m]['figure4'][k]['estimate'],k+' independent point');close(interval,S[m]['figure4'][k]['ci95'],k+' independent interval')
  print('Independent numerical verification:',m,flush=True)
 report={'passed':True,'independently_scored_chosen_attempts':checked_attempts,'independently_scored_novel_objects':checked_probes,'checks':['Raw endpoint Gaussian and exit scoring','All interaction, feedback, JSON, decomposition and transfer estimates and intervals','Primary and broad solve numerators','All Figure 4 conditional means and intervals','Figure 4 A/C via residual regression; B via full dummy-variable residualization; all 2000-draw intervals'],'tolerance':1e-9}
 (results/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=Path,default=ROOT);p.add_argument('--results',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.repo_root,a.results),indent=2))
