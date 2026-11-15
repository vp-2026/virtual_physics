"""Direct provider adapters. Credentials never enter prompts, logs, or files."""
import base64, hashlib, json, os, subprocess, time, urllib.request, urllib.error
from pathlib import Path
MODELS={
 'astra':('gpt-6-astra','openai',10.0,50.0),
 'opus':('claude-opus-5-5','anthropic',4.0,20.0),
 'robotics':('gemini-robotics-er-2-preview','google',1.0,5.0),
}
class DirectProvider:
 def __init__(self,key,system_prompt,output,cap):
  self.key=key;self.model,self.provider,self.input_price,self.output_price=MODELS[key]
  self.cloud_project=os.environ.get('VTOOLS_GCP_PROJECT') if key=='robotics' else None
  self._access_token=None;self._token_time=0
  if self.cloud_project:
   self.provider='google_cloud'
   # Cloud's advertised $1/$5 promotion is credits back. Reserve gross cost.
   self.input_price=2.0;self.output_price=10.0
  self.system_prompt=system_prompt;self.max_tokens=8192;self.cap=cap
  self.root=Path(output);self.root.mkdir(parents=True,exist_ok=True)
  self.ledger=self.root/'provider-usage.jsonl';self.cache=self.root/'provider-cache';self.cache.mkdir(exist_ok=True)
  self.cumulative_cost_usd=sum(json.loads(l)['estimated_cost_usd']for l in self.ledger.read_text().splitlines()) if self.ledger.exists() else 0.0
  self.last_call_metadata={};self.last_request_time=0
  env={'astra':'OPENAI_API_KEY','opus':'ANTHROPIC_API_KEY','robotics':'GEMINI_API_KEY'}[key]
  self.credential=os.environ.get(env)
  if not self.credential and key=='robotics' and not self.cloud_project:
   r=subprocess.run(['security','find-generic-password','-s','codex-gemini-api-key','-w'],capture_output=True,text=True)
   if r.returncode==0:self.credential=r.stdout.strip()
  if not self.credential and not self.cloud_project:raise RuntimeError(f'Missing {env}; enter it locally in Terminal, never in chat.')
 def cloud_access_token(self):
  if not self._access_token or time.monotonic()-self._token_time>=2700:
   try:r=subprocess.run(['gcloud','auth','print-access-token'],capture_output=True,text=True,timeout=45)
   except Exception:raise RuntimeError('Cloud authentication unavailable; checkpoint retained.') from None
   if r.returncode or not r.stdout.strip():raise RuntimeError('Cloud authentication needs renewal with gcloud auth login; checkpoint retained.')
   self._access_token=r.stdout.strip();self._token_time=time.monotonic()
  return self._access_token
 def generate(self,*,history,prompt,image_paths):
  if self.key!='opus' or os.environ.get('VTOOLS_OPUS_TOKEN_RECOVERY')!='1':
   return self._generate_once(history=history,prompt=prompt,image_paths=image_paths)
  base=self.max_tokens;multiplier=2 if 'Latest observable rollout JSON:' in prompt else 1
  caps=[];events=[];cap=base*multiplier
  try:
   while True:
    self.max_tokens=cap//multiplier;caps.append(cap)
    try:
     text=self._generate_once(history=history,prompt=prompt,image_paths=image_paths)
    except RuntimeError as error:
     metadata=dict(self.last_call_metadata)
     if str(error)!='Provider returned no final text; usage saved. Unit remains partial.' or metadata.get('finish_reason')!='max_tokens':raise
     events.append({'request_sha256':metadata.get('request_sha256'),'output_cap':cap,'empty_truncated':True,'cached_failure':metadata.get('recovered_cached_empty',False)})
     if cap>=32768:self._record_recovery(caps,events,prompt,image_paths,'still_partial');raise
     cap=min(cap*2,32768);continue
    if len(caps)>1:
     events.append({'request_sha256':self.last_call_metadata.get('request_sha256'),'output_cap':cap,'empty_truncated':False})
     recovery=self._record_recovery(caps,events,prompt,image_paths,'first_nonempty_response')
     self.last_call_metadata['token_budget_recovery']=recovery
    return text
  finally:self.max_tokens=base
 def _record_recovery(self,caps,events,prompt,image_paths,outcome):
  entry={'policy':'opus_empty_max_tokens_only_v1','authorized_date':'2026-11-02','maximum_output_tokens':32768,
   'unit':getattr(self,'current_unit',None),'phase':getattr(self,'current_phase',None),'request_caps':caps,'requests':events,
   'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'image_sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in image_paths],
   'outcome':outcome,'recorded_unix':time.time(),'paper_tables_unchanged':True}
  with (self.root/'token-budget-recovery.jsonl').open('a')as f:f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
  return entry
 def _generate_once(self,*,history,prompt,image_paths):
  limit=self.max_tokens*(2 if 'Latest observable rollout JSON:' in prompt else 1)
  encoded=[base64.b64encode(Path(p).read_bytes()).decode()for p in image_paths]
  messages=[{'role':'assistant' if m['role']=='assistant' else 'user','content':str(m.get('content',''))}for m in history]
  if self.key=='astra':
   content=[{'type':'input_text','text':prompt}]+[{'type':'input_image','image_url':'data:image/png;base64,'+b,'detail':'high'}for b in encoded]
   payload={'model':self.model,'instructions':self.system_prompt,'input':messages+[{'role':'user','content':content}],
     'max_output_tokens':limit,'reasoning':{'effort':'medium'},'text':{'format':{'type':'json_object'}},'store':False}
   url='https://api.openai.com/v1/responses';headers={'Authorization':'Bearer '+self.credential}
  elif self.key=='opus':
   content=[{'type':'text','text':prompt}]+[{'type':'image','source':{'type':'base64','media_type':'image/png','data':b}}for b in encoded]
   payload={'model':self.model,'system':self.system_prompt,'messages':messages+[{'role':'user','content':content}],
     'max_tokens':limit,'output_config':{'effort':'medium'}}
   url='https://api.anthropic.com/v1/messages';headers={'x-api-key':self.credential,'anthropic-version':'2023-06-01'}
  else:
   contents=[{'role':'model' if m['role']=='assistant' else 'user','parts':[{'text':m['content']}]}for m in messages]
   parts=[{'text':prompt}]+[{'inlineData':{'mimeType':'image/png','data':b}}for b in encoded]
   payload={'systemInstruction':{'parts':[{'text':self.system_prompt}]},'contents':contents+[{'role':'user','parts':parts}],
     'generationConfig':{'temperature':0.2,'maxOutputTokens':limit,'responseMimeType':'application/json','thinkingConfig':{'thinkingLevel':'MEDIUM','includeThoughts':False}}}
   url=f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent';headers={'x-goog-api-key':self.credential}
   if self.cloud_project:
    url=f'https://aiplatform.googleapis.com/v1/projects/{self.cloud_project}/locations/global/publishers/google/models/{self.model}:generateContent';headers={}
  raw=json.dumps(payload,separators=(',',':')).encode()
  # Keep original Developer API cache keys, but never substitute a response
  # from that route for a new Cloud request. Completed checkpoints are reused.
  cache_material=(url.encode()+b'\n'+raw) if self.cloud_project else raw
  digest=hashlib.sha256(cache_material).hexdigest();cached=self.cache/(digest+'.json')
  if cached.exists():
   saved=json.loads(cached.read_text());self.last_call_metadata={**saved['metadata'],'estimated_cost_usd':0,'recovered_cached_response':True};return saved['text']
  empty_cache=cached.with_suffix('.empty.json')
  if self.key=='opus' and os.environ.get('VTOOLS_OPUS_TOKEN_RECOVERY')=='1' and empty_cache.exists():
   self.last_call_metadata={**json.loads(empty_cache.read_text()),'estimated_cost_usd':0,'recovered_cached_empty':True}
   raise RuntimeError('Provider returned no final text; usage saved. Unit remains partial.')
  # Conservative reservation: one token per UTF-8 byte for text, 4096 per 600px
  # image, and the entire output cap. Actual provider usage is the spend ledger.
  text_bytes=len((self.system_prompt+prompt+json.dumps(messages)).encode())
  reserve=((text_bytes+len(encoded)*4096)*self.input_price+limit*self.output_price)/1e6
  if self.cumulative_cost_usd+reserve>self.cap:raise RuntimeError('Budget reservation reached; checkpoint retained as partial, not a benchmark failure.')
  started=time.time()
  for retry in range(4):
   if self.key=='robotics':time.sleep(max(0,(2 if self.cloud_project else 10)-(time.time()-self.last_request_time)))
   if self.cloud_project:headers={'Authorization':'Bearer '+self.cloud_access_token()}
   request=urllib.request.Request(url,data=raw,headers={'Content-Type':'application/json',**headers},method='POST')
   self.last_request_time=time.time()
   try:
    with urllib.request.urlopen(request,timeout=300)as response:result=json.load(response)
    break
   except urllib.error.HTTPError as e:
    # Keep only quota metrics/limits, never raw provider bodies or resource IDs.
    try:err=json.loads(e.read()).get('error',{});kind=err.get('type') or err.get('status') or 'provider_error';param=err.get('param')
    except Exception:err={};kind='provider_error';param=None
    quota=[];delay=60
    for detail in err.get('details',[]):
     if 'retryDelay' in detail:
      try:delay=max(delay,float(detail['retryDelay'].rstrip('s')))
      except ValueError:pass
     for v in detail.get('violations',[]):quota.append({k:v[k]for k in ('quotaMetric','quotaId','quotaValue')if k in v})
    (self.root/'last-provider-error.json').write_text(json.dumps({'http_status':e.code,'type':kind,'param':param,'quota':quota,'retry_delay_seconds':delay}))
    daily=any('perday' in str(q).lower() for q in quota)
    if e.code==401 and self.cloud_project and retry==0:
     self._access_token=None;continue
    if e.code==429 and retry<3 and not daily:
     time.sleep(min(delay+2,120));continue
    raise RuntimeError(f'{self.provider} HTTP {e.code}; type={kind}; param={param}; quota details saved privately') from None
   except Exception as error:
    # Record only exception types and numeric errno, never URLs, headers or keys.
    detail={'exception_type':type(error).__name__,'errno':getattr(error,'errno',None),'reason_type':type(getattr(error,'reason',None)).__name__,'received_unix':time.time()}
    (self.root/'last-transport-error.json').write_text(json.dumps(detail))
    raise RuntimeError(f'{self.provider} request interrupted ({type(error).__name__}); no automatic retry of an ambiguously billed call.') from None
  if self.key=='astra':
   text='\n'.join(c.get('text','')for item in result.get('output',[])for c in item.get('content',[])if c.get('type')=='output_text')
   usage=result.get('usage')or{};n_in=usage.get('input_tokens',0);n_out=usage.get('output_tokens',0);finish=result.get('status');returned=result.get('model')
  elif self.key=='opus':
   text='\n'.join(c.get('text','')for c in result.get('content',[])if c.get('type')=='text');usage=result.get('usage')or{}
   n_in=usage.get('input_tokens',0)+usage.get('cache_creation_input_tokens',0)+usage.get('cache_read_input_tokens',0);n_out=usage.get('output_tokens',0);finish=result.get('stop_reason');returned=result.get('model')
  else:
   candidate=(result.get('candidates')or[{}])[0];text='\n'.join(p.get('text','')for p in candidate.get('content',{}).get('parts',[])if not p.get('thought'))
   usage=result.get('usageMetadata')or{};n_in=usage.get('promptTokenCount',0);n_out=usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0);finish=candidate.get('finishReason');returned=result.get('modelVersion')
  n_cached=min(n_in,usage.get('cachedContentTokenCount',0)) if self.cloud_project else 0
  cost=((n_in-n_cached)*self.input_price+n_cached*self.input_price*.1+n_out*self.output_price)/1e6;self.cumulative_cost_usd+=cost
  self.last_call_metadata={'provider':self.provider,'requested_model':self.model,'returned_model':returned,'usage':usage,
   'estimated_cost_usd':cost,'cumulative_cost_usd':self.cumulative_cost_usd,'image_count':len(encoded),'finish_reason':finish,
   'received_unix':time.time(),'duration_seconds':time.time()-started,'request_sha256':digest,'request_parameters':{'max_output_tokens':limit,'reasoning_effort':'medium','seed_supported':False,'temperature':0.2 if self.key=='robotics' else None},
   'price_assumption_usd_per_million':{'input':self.input_price,'output':self.output_price},'response_id':result.get('id')or result.get('responseId')}
  if self.cloud_project:
   self.last_call_metadata.update({'endpoint_region':'global','price_basis':'gross_before_promotional_credits',
    'price_source':'https://cloud.google.com/model-garden/pricing','cached_input_tokens':n_cached,
    'cached_input_price_usd_per_million':self.input_price*.1})
  with self.ledger.open('a')as f:f.write(json.dumps(self.last_call_metadata)+'\n');f.flush();os.fsync(f.fileno())
  if not text:
   if self.key=='opus' and finish=='max_tokens':
    with empty_cache.open('w')as f:os.chmod(empty_cache,0o600);f.write(json.dumps(self.last_call_metadata));f.flush();os.fsync(f.fileno())
   raise RuntimeError('Provider returned no final text; usage saved. Unit remains partial.')
  temporary=cached.with_suffix('.json.tmp')
  with temporary.open('w')as f:
   os.chmod(temporary,0o600);f.write(json.dumps({'text':text,'metadata':self.last_call_metadata}));f.flush();os.fsync(f.fileno())
  temporary.replace(cached)
  return text
