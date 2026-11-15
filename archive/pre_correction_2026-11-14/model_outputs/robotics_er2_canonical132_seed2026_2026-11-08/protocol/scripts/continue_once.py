"""Bounded continuation of an existing canonical run, not a recurring automation."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--model',choices=['astra','opus','robotics'],required=True);args=p.parse_args();out=ROOT/'private'/args.model
lock=(out/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=json.loads((out/'canonical132-manifest.json').read_text())
env=os.environ.copy()
if args.model=='robotics' and not env.get('VTOOLS_GCP_PROJECT'):raise SystemExit('Set VTOOLS_GCP_PROJECT to your own project before running the cloud continuation.')
for attempt in range(3):
 status=json.loads((out/'status.json').read_text())
 if status['state']=='complete':break
 if status['state']=='needs_attention'and not all('request interrupted'in x['reason']for x in status.get('partial_units',[])):break
 pidfile=out/'canonical132.pid'
 if pidfile.exists():
  result=subprocess.run(['ps','-p',pidfile.read_text().strip(),'-o','command='],capture_output=True,text=True)
  if 'scripts/run.py'in result.stdout:raise SystemExit('Existing worker retained; duplicate launch prevented.')
 with(out/'continuation-events.jsonl').open('a')as file:file.write(json.dumps({'time':time.time(),'resume':attempt+1,'prior_status':status,'possible_unrecorded_request_cost':True})+'\n')
 with(out/'canonical132.log').open('a')as log:
  child=subprocess.Popen([sys.executable,str(ROOT/'scripts/run.py'),'--model',args.model,'--scope','canonical132','--cap-usd',str(manifest['spend_cap_usd'])],cwd=ROOT,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
  pidfile.write_text(str(child.pid)+'\n');code=child.wait()
 if code==0:break
 time.sleep(15)
(out/'continuation-result.json').write_text(json.dumps({'finished_unix':time.time(),'launches':attempt+1,'final_state':json.loads((out/'status.json').read_text())['state']}))
