"""Historical archive joins and probe scoring, recovered without private loaders."""
import copy, math
import numpy as np
def mean(x):
    return float(np.mean(x)) if len(x) else None

def point(p):
 try:return [int(round(float(x)))for x in p]
 except (TypeError,ValueError):return None

def archived_attempt(record,condition,number,coords):
 prefix=('branches/trace_status/independent_attempt_1/'if condition=='trace_status'else'shared_attempt_1/')if number==1 else f'branches/{condition}/'
 found=[a for a in record['attempt_records']if a['source'].startswith(prefix)and a['attempt']==number and a['coords']==coords and not a.get('obstruction_detected')]
 assert len(found)==1,('attempt reconstruction',record['source_unit'],condition,number,len(found))
 a=copy.deepcopy(found[0]);label='shared_attempt_1'if number==1 else f'{condition}_attempt_{number}'
 calls=[c for c in record['provider_calls']if c['source'].startswith(prefix)and c['call_label']==label and point(((c.get('parsed_action_payload')or{}).get('parsed')or{}).get('point'))==coords]
 assert calls,('prediction reconstruction',record['source_unit'],condition,number)
 a['model_response']=calls[-1]['parsed_action_payload']
 return a

def side_scores(item):
 rows=item.get('score_rows')or[]
 if not rows:return None
 out={'coverage':mean([bool(r['prediction_present'])for r in rows]),'state_accuracy':mean([bool(r['state_correct'])for r in rows])}
 for sigma in [75,100,150]:
  scores=[]
  for r in rows:
   if r['predicted_state']=='in_scene'and r['actual_state']=='in_scene'and r.get('endpoint_error_px')is not None:
    score=math.exp(-float(r['endpoint_error_px'])**2/(2*sigma*sigma))
   else:score=float(r['prediction_quality_sigma150'])
   scores.append(score)
  out[f'pq{sigma}']=mean(scores)
 assert abs(out['pq150']-mean([r['prediction_quality_sigma150']for r in rows]))<1e-10
 return out
