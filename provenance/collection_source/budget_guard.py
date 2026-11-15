"""Cross-process USD reservations; no credential or prompt content is logged."""
import fcntl,json,time,uuid,os,shutil
from pathlib import Path
class Guard:
 def __init__(self,provider,root,group,cap,prices,baseline=0):
  self.p=provider;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.group=group;self.cap=cap;self.prices=prices;self.baseline=baseline
  self.lock=self.root/(group+'.lock');self.state=self.root/(group+'.json')
 def update(self,fn):
  with self.lock.open('a')as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   d=json.loads(self.state.read_text())if self.state.exists()else {'spent_usd':self.baseline,'pending':{},'completed_calls':0,'cap_usd':self.cap,'next_request_time':0}
   result=fn(d);tmp=self.state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(self.state);return result
 def generate(self,*,history,prompt,image_paths):
  if shutil.disk_usage(self.root).free<2*1024**3:raise RuntimeError('Less than 2 GB free; no request submitted')
  text=len((self.p.system_prompt+prompt+json.dumps(history)).encode())
  limit=self.p.max_tokens*(2 if 'Latest observable rollout JSON:'in prompt else 1)
  reserve=((text+4096*len(image_paths))*self.prices[0]+limit*self.prices[1])/1e6
  if getattr(self.p,'token_recovery',False):
   caps=list(dict.fromkeys([limit,min(limit*2,32768),min(limit*4,32768)]))
   reserve=((text+4096*len(image_paths))*len(caps)*self.prices[0]+sum(caps)*self.prices[1])/1e6
  # The original OpenRouter transport may make up to four attempts.
  if self.group=='openrouter':reserve*=4
  token=uuid.uuid4().hex
  def start(d):
   if d['spent_usd']+sum(d['pending'].values())+reserve>self.cap:raise RuntimeError('Global cost reservation cap reached; checkpoint retained')
   d['pending'][token]=reserve
   delay=max(0,d['next_request_time']-time.time())if self.group=='robotics'else 0
   if self.group=='robotics':d['next_request_time']=time.time()+delay+10
   return delay
  delay=self.update(start)
  if delay:time.sleep(delay)
  try:
   value=self.p.generate(history=history,prompt=prompt,image_paths=image_paths)
   md=self.p.last_call_metadata;reported=md.get('logical_call_billed_cost_usd',(md.get('usage')or{}).get('cost'))
   if reported is None:reported=md.get('estimated_cost_usd')
   cost=float(reported)if reported is not None else reserve
   extra=md.get('response_extra')or{}
   if extra.get('provider'):md['backend_provider']=extra['provider']
  except Exception:
   # Conservatively charge the reservation when billing cannot be determined.
   def fail(d):d['spent_usd']+=d['pending'].pop(token);d['completed_calls']+=1
   self.update(fail);raise
  def finish(d):d['pending'].pop(token);d['spent_usd']+=cost;d['completed_calls']+=1
  self.update(finish);return value
 def __getattr__(self,k):return getattr(self.p,k)
