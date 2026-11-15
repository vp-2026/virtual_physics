"""Launch/resume only the frozen corrected-scene worklist; default is offline."""
from pathlib import Path
import argparse,json,os,subprocess,concurrent.futures,urllib.request
O=Path(__file__).resolve().parent
PY='python3'
p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',choices=['gpt5','gemini31pro','qwen36plus','robotics_er2'],required=True);p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--exclude-indices',nargs='*',type=int,default=[]);a=p.parse_args()
work=json.loads((O/'preflight/worklist.json').read_text());checks={r['index']:r['passed']for r in json.loads((O/'preflight/replay_validation.json').read_text())};todo=[]
for i,w in enumerate(work):
 if w['model']not in a.models:continue
 if i in a.exclude_indices:continue
 assert checks[i]
 s=O/'status'/f'{i:03d}.json'
 if s.exists()and json.loads(s.read_text())['phase']=='complete':continue
 todo.append(i)
todo.sort(key=lambda i:(work[i]['goal_id'],work[i]['model']))
print(json.dumps({'queued_units':len(todo),'models':a.models,'execute':a.execute,'workers':a.workers,'total_experiment_cap_usd':198,'separate_vqa_cap_usd':2}),flush=True)
if not a.execute:raise SystemExit(0)
if any(m!='robotics_er2'for m in a.models):
 k=(Path.home()/'.config/openrouter/api_key').read_text().strip()
 if k.startswith(('OPENROUTER_API_KEY=','export OPENROUTER_API_KEY=')):k=k.split('=',1)[1].strip().strip('\"\'')
 try:
  with urllib.request.urlopen(urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+k}),timeout=30)as f:json.load(f)
 except Exception:raise RuntimeError('OpenRouter credential preflight failed; no inference queued')from None
 del k
if 'robotics_er2'in a.models:
 err=O/'providers/robotics_er2/last-provider-error.json'
 if err.exists():
  d=json.loads(err.read_text())
  if any('PerDay' in str(q.get('quotaId','')) for q in d.get('quota',[]))and not os.environ.get('GEMINI_API_KEY')and not os.environ.get('VTOOLS_GCP_PROJECT'):
   raise RuntimeError('Recorded daily Gemini quota exhaustion. Supply an authorized credential/project before resuming; no requests queued.')
env=dict(os.environ);env['PYTHONPATH']=str(O/'runtime')+':<SIMULATOR_PYTHON_PACKAGES>';env['OPENBLAS_NUM_THREADS']='1';logs=O/'logs';logs.mkdir(exist_ok=True)
def run(i):
 with (logs/f'{i:03d}.log').open('a')as f:
  r=subprocess.run([PY,str(O/'worker.py'),'--index',str(i),'--execute','--cap-usd','10'],stdout=f,stderr=subprocess.STDOUT,env=env)
 s=O/'status'/f'{i:03d}.json';status=json.loads(s.read_text())if s.exists()else {'phase':'needs_attention'}
 return {'index':i,'model':work[i]['model'],'goal_id':work[i]['goal_id'],'exit_code':r.returncode,'phase':status['phase']}
# Bounded continuous queue. Keep independent tasks moving if a response exhausts
# its predeclared format/placement repair allowance; never score it as failure.
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)as pool:
 queue=iter(todo);pending={};stop=False
 for i in [next(queue,None)for _ in range(a.workers)]:
  if i is not None:pending[pool.submit(run,i)]=i
 while pending:
  done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
  for f in done:
   pending.pop(f);r=f.result();print(json.dumps(r),flush=True)
   if r['exit_code']:
    status_path=O/'status'/f"{r['index']:03d}.json";detail=json.loads(status_path.read_text()).get('error','')if status_path.exists()else''
    if any(s in detail.lower()for s in ['cost','budget','401','403','quota','authentication','less than 2 gb']):stop=True
   if not stop:
    i=next(queue,None)
    if i is not None:pending[pool.submit(run,i)]=i
 if stop:raise SystemExit('Provider/storage/budget issue; stopped new submissions, retained checkpoints.')
