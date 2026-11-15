"""The frozen November 2 empty-length-only Qwen recovery policy."""
import copy,hashlib,json,time
from pathlib import Path
def provider_class(base,unit):
 class Recovery(base):
  token_recovery=True
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.client.max_retries=0;self.unit=Path(unit);self._logical_cost=0.;self.prior_empty=[]
   for path in self.unit.rglob('provider_calls.jsonl'):
    for line in path.read_text().splitlines():
     row=json.loads(line);md=row.get('provider_metadata')or{}
     if not row.get('response_text')and md.get('finish_reason')=='length':self.prior_empty.append(row)
  def _request_with_retries(self,kwargs):
   request=copy.deepcopy(kwargs);initial=int(request['max_tokens']);limit=initial;events=[];cost=0.;start_limit=initial
   content=request['messages'][-1]['content'];prompt=next(x['text']for x in content if x.get('type')=='text');images=[]
   import base64
   for x in content:
    if x.get('type')=='image_url':images.append(hashlib.sha256(base64.b64decode(x['image_url']['url'].split(',',1)[1])).hexdigest())
   # A stopped request can resume its exact saved prompt/images instead of paying
   # for another known-empty base-cap call. The surrounding history is rebuilt
   # from the unchanged last valid branch checkpoint by the original runner.
   prior=next((r for r in self.prior_empty if r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()and r['image_sha256']==images and int(r['provider_metadata']['request_parameters']['max_tokens'])==initial),None)
   if prior:
    limit=min(initial*2,32768);events.append({'max_tokens':initial,'reused_saved_empty_length_response':True,'source_response_sha256':prior.get('response_sha256'),'usage':prior['provider_metadata'].get('usage')});self.prior_empty.remove(prior)
   for attempt in range(3 if not prior else 2):
    request['max_tokens']=limit;started=time.time();response=super()._request_with_retries(request);usage=response.usage.model_dump(mode='json')if response.usage else {};cost+=float(usage.get('cost')or 0)
    choice=response.choices[0];text=choice.message.content or ''
    events.append({'max_tokens':limit,'finish_reason':choice.finish_reason,'response_text':text,'usage':usage,'returned_model':response.model,'duration_seconds':time.time()-started})
    raw={'policy':'empty_length_truncation_only_v1','timestamp_unix':time.time(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'history_sha256':hashlib.sha256(json.dumps(kwargs['messages'][:-1],sort_keys=True).encode()).hexdigest(),'image_sha256':images,**events[-1]}
    self.unit.mkdir(parents=True,exist_ok=True)
    with (self.unit/'token_recovery_raw.jsonl').open('a')as stream:stream.write(json.dumps(raw)+'\n');stream.flush()
    if text or choice.finish_reason!='length' or limit>=32768:break
    limit=min(2*limit,32768)
   self._logical_cost=cost;self._recovery_events=events;self._actual_limit=limit
   if len(events)>1:
    row={'policy':'empty_length_truncation_only_v1','base_output_tokens':initial,'maximum_output_tokens':32768,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'history_sha256':hashlib.sha256(json.dumps(kwargs['messages'][:-1],sort_keys=True).encode()).hexdigest(),'image_sha256':images,'events':events,'new_billed_cost_usd':cost,'timestamp_unix':time.time()}
    path=self.unit/'token_budget_recovery_calls.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
   return response
  def generate(self,**kwargs):
   value=super().generate(**kwargs);md=self.last_call_metadata;last=float((md.get('usage')or{}).get('cost')or 0);self.cumulative_cost_usd+=self._logical_cost-last
   md['logical_call_billed_cost_usd']=self._logical_cost;md['request_parameters']['max_tokens']=self._actual_limit
   if len(self._recovery_events)>1:md['token_budget_recovery']={'policy':'empty_length_truncation_only_v1','attempt_caps':[x['max_tokens']for x in self._recovery_events],'first_nonempty_retained':True}
   return value
 return Recovery
