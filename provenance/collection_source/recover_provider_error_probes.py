"""One reviewed technical restart of explicitly errored prediction requests."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
O=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--indices',nargs='*',type=int);p.add_argument('--execute',action='store_true');a=p.parse_args();W=json.loads((O/'preflight/worklist.json').read_text());prepared=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for i,w in enumerate(W):
 if a.indices and i not in a.indices:continue
 if w['selection']is None:continue
 status=O/'status'/f'{i:03d}.json';s=json.loads(status.read_text())if status.exists()else{}
 if s.get('phase')!='complete':continue
 unit=Path(s['unit_dir']);record=unit/'probe_provider_error_recovery.json'
 if record.exists():continue
 failed=[]
 for f in unit.glob('transfer_sidecars/**/result.json'):
  r=json.loads(f.read_text());md=r.get('last_provider_metadata')or{}
  if not r.get('schema_valid')and md.get('finish_reason')=='error':failed.append(f)
 if not failed:continue
 print(i,w['goal_id'],'provider-error probes',len(failed),flush=True)
 if not a.execute:continue
 with(O/'locks'/f'{i:03d}.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  note={'index':i,'goal_id':w['goal_id'],'model':w['model'],'policy':'One checkpoint restart only for invalid final responses explicitly marked provider error. No valid, ordinary malformed, or token-limited completed response is resampled.','started_unix':time.time(),'main_summary_sha256':sha(unit/'unit_summary.json'),'main_call_sha256':{str(f.relative_to(unit)):sha(f)for f in unit.rglob('provider_calls.jsonl')if 'transfer_sidecars'not in f.parts},'queries':[]}
  for f in failed:
   backup=f.with_name('result.provider_error_before_recovery.json');assert not backup.exists();note['queries'].append({'path':str(f.relative_to(unit)),'original_sha256':sha(f),'backup':str(backup.relative_to(unit))});f.rename(backup)
  summary=unit/'transfer_sidecars/summary.json';backup=summary.with_name('summary.provider_error_before_recovery.json');assert not backup.exists();summary.rename(backup);note['original_summary_sha256']=sha(backup)
  record.write_text(json.dumps(note,indent=2)+'\n');s.update(phase='needs_attention',error='Reviewed provider-error probe recovery; original errors preserved.',updated_unix=time.time());status.write_text(json.dumps(s,indent=2)+'\n');prepared.append((i,unit))
if not a.execute:raise SystemExit(0)
env=dict(os.environ);env['PYTHONPATH']=str(O/'runtime')+':<SIMULATOR_PYTHON_PACKAGES>';env['OPENBLAS_NUM_THREADS']='1';PY='python3'
def run(pair):
 i,unit=pair
 with(O/'logs'/f'{i:03d}.log').open('a')as log:r=subprocess.run([PY,str(O/'worker.py'),'--index',str(i),'--execute','--resume-error','--cap-usd','10'],stdout=log,stderr=subprocess.STDOUT,env=env)
 note=json.loads((unit/'probe_provider_error_recovery.json').read_text());assert sha(unit/'unit_summary.json')==note['main_summary_sha256']
 for rel,h in note['main_call_sha256'].items():assert sha(unit/rel)==h
 unresolved=[]
 for q in note['queries']:
  f=unit/q['path']
  if not f.exists():unresolved.append(q['path']);continue
  value=json.loads(f.read_text());md=value.get('last_provider_metadata')or{};q.update(replacement_sha256=sha(f),schema_valid=value.get('schema_valid'),finish_reason=md.get('finish_reason'))
  if not value.get('schema_valid')and md.get('finish_reason')=='error':unresolved.append(q['path'])
 note.update(finished_unix=time.time(),unresolved_queries=unresolved,exit_code=r.returncode,main_records_unchanged=True);(unit/'probe_provider_error_recovery.json').write_text(json.dumps(note,indent=2)+'\n')
 if unresolved:
  status=O/'status'/f'{i:03d}.json';s=json.loads(status.read_text());s.update(phase='needs_attention',error='Provider-error probes remain missing after one technical restart',probe_gaps=unresolved,unit_dir=str(unit));status.write_text(json.dumps(s,indent=2)+'\n')
 print(json.dumps({'index':i,'recovered_queries':len(note['queries'])-len(unresolved),'unresolved':len(unresolved),'exit_code':r.returncode}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5)as pool:list(pool.map(run,prepared))
