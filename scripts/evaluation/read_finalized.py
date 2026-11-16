"""Read and score the finalized model records; no model calls or human data."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from record_scoring import recorded_attempt,side_scores
import analyze_action_predictions as scorer
MODELS=('gpt5','gemini31pro','qwen36plus','robotics_er2')
def read_units(repo,output):
 root=repo/'data/finalized';manifest=json.loads((root/'manifest.json').read_text());data={m:{}for m in MODELS}
 for shard in manifest['shards']:
  path=root/shard['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==shard['sha256'];model=shard['model'];count=0
  with gzip.open(path,'rt')as f:
   for line in f:
    record=json.loads(line);s=record['summary'];g=s['goal'];gid=g['balanced_goal_id'];assert gid not in data[model];count+=1
    u={'goal':gid,'layout':g['puzzle_key'].rsplit('_',1)[0],'upward':int(g['condition']=='upward'),'seed':s['seed'],'arms':{},'transfer':{},'issues':[]}
    assert u['seed']==2026
    for c,b in s['conditions'].items():
     attempts=[]
     for n,coords in enumerate(b['coords'],1):
      try:
       a=recorded_attempt(record,c,n,coords);scores=scorer.score_attempt(summary=s,unit_dir=output,condition=c,attempt=a);assert scores
       attempts.append(dict(attempt=n,coords=coords,S=int(a['goal_succeeded']),**{f'pq{q}':float(np.mean([v[f'prediction_quality_sigma{q}']for v in scores]))for q in (75,100,150)}))
      except (AssertionError,KeyError)as e:u['issues'].append({'condition':c,'attempt':n,'error':type(e).__name__})
     u['arms'][c]={'solved':int(b['solved_by_budget']),'attempts':attempts,'count':b['attempt_count']}
    operational_errors={v['source']for v in record['sidecar_records']if v['source'].endswith('/result.json')and not v['record'].get('schema_valid')and (v['record'].get('last_provider_metadata')or{}).get('finish_reason')=='error'}
    u['provider_error_probe_sources']=sorted(operational_errors)
    sides=[v['record']for v in record['sidecar_records']if v['source']=='transfer_sidecars/summary.json'];assert len(sides)<=1
    if sides:
     side=sides[0];pre=side['results']['pre_feedback'];sel=side['probe_selection'];u['early_probe_coords']=sel['early_probe']['coords'];u['terminal_probe_coords']=sel['terminal_probe']['coords']
     for c,b in side['results']['conditions'].items():
      rep='trace'if c=='trace_status'else'visual';u['transfer'][c]={'contaminated':bool(b['terminal_probe_contaminated']),'terminal_pre':side_scores(pre[f'{rep}_terminal_probe']),'terminal_post':side_scores(b['terminal_prediction']),'early_pre':side_scores(pre[f'{rep}_early_probe']),'early_post':side_scores(b['early_prediction'])}
      paths={'terminal_pre':f'pre_feedback/{rep}/terminal_probe','early_pre':f'pre_feedback/{rep}/early_probe','terminal_post':f'{c}/terminal_prediction','early_post':f'{c}/early_prediction'}
      missing=[k for k,v in paths.items()if 'transfer_sidecars/'+v+'/result.json'in operational_errors];u['transfer'][c]['provider_error_moments']=missing
      for k in missing:u['transfer'][c][k]=None
    data[model][gid]=u
  assert count==shard['records']
 return data,manifest

def transfer_issues(u,conditions=('full','neither','trace_status')):
 errors=[]
 for c in conditions:
  t=u['transfer'].get(c);arm=u['arms'].get(c)
  if not t or not arm:errors.append(c+':missing');continue
  if t['contaminated']:errors.append(c+':terminal_probe_within_40px_of_executed_action')
  errors.extend(c+':provider_error_'+k for k in t.get('provider_error_moments',[]))
  if any(t.get(moment)is None for moment in ['terminal_pre','terminal_post','early_pre','early_post']):errors.append(c+':missing_score')
  if arm['attempts'] and np.linalg.norm(np.asarray(u['early_probe_coords'])-arm['attempts'][0]['coords'])<40:errors.append(c+':early_probe_within_40px_of_first_action')
 return errors
