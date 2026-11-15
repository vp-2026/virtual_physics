#!/usr/bin/env python3
"""Execute the frozen current-scene battery with a conservative USD budget.

Only transport failures may be retried (at most once); valid but incorrect or
malformed answers are never selected away. Raw API responses and exact request
bodies are retained. No credentials are saved. Default mode is a free dry run.
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
import run_vqa as v
from generate_vqa import sha256_file, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--battery',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pricing',type=Path,required=True)
    p.add_argument('--budget-usd',type=float,default=10);p.add_argument('--execute',action='store_true')
    a=p.parse_args();b=a.battery.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=v.read_json(b/'freeze_sha256.json')
    for path,h in frozen.items():
        if sha256_file(b/path)!=h:raise ValueError('Frozen test-set checksum mismatch: '+path)
    prices={x['id']:x['pricing'] for x in v.read_json(a.pricing)}
    calls=v.read_jsonl(b/'call_manifest.jsonl');ids=v.DEFAULT_OPENROUTER_MODELS
    settings=v.read_json(b/'protocol.json')['model_settings']
    max_tokens=settings['max_tokens'];effort=settings['reasoning_effort']
    reserve={m:10000*float(prices[mid]['prompt'])+max_tokens*float(prices[mid]['completion']) for m,mid in ids.items()}
    estimate=sum(reserve[c['model_key']] for c in calls)
    print(json.dumps(dict(calls=len(calls),conservative_run_estimate_usd=estimate,budget_usd=a.budget_usd,execute=a.execute)),flush=True)
    if not a.execute:return
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:raise RuntimeError('OPENROUTER_API_KEY is required')
    # Free authentication check before queuing any inference requests.
    request=urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':f'Bearer {key}'})
    try:
        with urllib.request.urlopen(request,timeout=30) as response:
            json.load(response)
    except Exception:
        raise RuntimeError('OpenRouter credential preflight failed; no inference queued') from None
    logfile=out/'responses.jsonl'
    history=v.read_jsonl(logfile) if logfile.exists() else []
    # Reserve worst-case estimated charges even for failed requests whose usage
    # cannot be recovered. Every new attempt consumes another reservation.
    def charge(x):
        if str(x.get('error','')).startswith(('HTTP 401:', 'HTTP 403:')):return 0.0
        cost=(x.get('usage')or{}).get('cost')
        return float(cost) if cost is not None else reserve[x['model_key']]
    spent=sum(charge(x) for x in history)
    counts={c['call_id']:sum(x['call_id']==c['call_id'] for x in history) for c in calls}
    complete={r['call_id'] for r in history if 'error' not in r}
    todo=[c for c in calls if c['call_id'] not in complete and counts[c['call_id']]<2]
    def run(c):
        attempt=counts[c['call_id']]+1
        payload=v.read_json(b/c['payload_path']);payload['image_path']=str(b/payload['image_path'])
        body=v.build_request(payload,ids[c['model_key']],'openrouter',2026,effort,max_tokens)
        # A separately frozen interface diagnostic may request JSON-object
        # mode. It never replaces the primary strict-schema condition.
        if settings.get('response_format') == 'json_object':
            body['response_format']={'type':'json_object'}
        elif settings.get('response_format','json_schema') != 'json_schema':
            raise ValueError('Unsupported frozen response format')
        tag=f"{c['model_key']}_{c['layout_id']}_{attempt}"
        write_json(out/'requests'/(tag+'.json'),body)
        start=time.time()
        common={**c,'model_id_requested':ids[c['model_key']],'seed':2026,'reasoning_effort':effort,
            'max_tokens':max_tokens,'request_file':f'requests/{tag}.json','attempt':attempt}
        try:
            raw,_=v.request_json(v.ENDPOINTS['openrouter'],key,body,'openrouter',0,240,2026)
            write_json(out/'raw'/(tag+'.json'),raw)
            text=v.extract_text(raw);parsed,errors=v.parse_answer(text,payload)
            row={**common,'model_id_returned':raw.get('model'),'response_text':text,
                'parsed_response':parsed,'parse_errors':errors,'format_valid':not errors,
                'usage':raw.get('usage'),'raw_file':f'raw/{tag}.json'}
        except Exception as exc:
            row={**common,'error':v.sanitized_error_detail(str(exc),key),'format_valid':False}
        row.update(completed_at_unix=time.time(),latency_seconds=time.time()-start)
        return row
    # Reserve estimated worst-case charges for every in-flight request, then
    # settle with returned provider cost. Unknown charges retain the reserve.
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        pending={};queue=iter(todo);next_call=next(queue,None)
        while next_call is not None or pending:
            while next_call is not None and len(pending)<12:
                if shutil.disk_usage(out).free < 512*1024**2:
                    break
                cost=reserve[next_call['model_key']]
                if spent+sum(pending.values())+cost>a.budget_usd:break
                pending[pool.submit(run,next_call)]=cost;next_call=next(queue,None)
            if not pending:
                print('Budget or local-storage reserve exhausted; remaining calls not submitted.',flush=True);break
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future);row=future.result();spent+=charge(row)
                v.append_jsonl(logfile,row)
                print(json.dumps({k:row.get(k) for k in ['model_key','layout_id','format_valid','error']}),flush=True)
                if str(row.get('error','')).startswith(('HTTP 401:', 'HTTP 403:')):
                    raise RuntimeError('Authentication failed during run; stopped submitting calls')
    history=v.read_jsonl(logfile)
    write_json(out/'run_accounting.json',dict(requests=len(history),reserved_upper_estimate_usd=spent,
        budget_usd=a.budget_usd,reported_cost_usd=sum(float((r.get('usage')or{}).get('cost')or 0) for r in history),
        usage_missing_requests=sum(not r.get('usage') for r in history),
        reservation_assumptions=f'10,000 input + {max_tokens} output tokens per pending request at recorded prices; settle known actual cost, retain reservation for unknown charges.'))


if __name__=='__main__':main()
