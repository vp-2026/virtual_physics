#!/usr/bin/env python3
"""Export the completed ER2 extension without running a model or simulator.

The source root contains canonical132-manifest.json and episodes/. Cached
records are private staging artifacts. Re-run with the same source to resume.
"""
import argparse,collections,concurrent.futures,csv,gzip,hashlib,json,os,re,subprocess,time
from pathlib import Path
import export_model_outputs as ex
from validate_documented_snapshot import terminal,sidecar_complete,scan

MODEL='gemini-robotics-er-2-preview'
SNAPSHOT='robotics_er2_canonical132_seed2026_2026-11-08'
SKIP={'assets','rollout_frames','truth_trace','goal_transfer','__pycache__'}
PRIVATE_TERMS=[]
def walk(root,names):
    out=[]
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in SKIP]
        out.extend(Path(directory)/f for f in files if f in names)
    return sorted(out)

def private_clean(value):
    value=ex.sanitize(value)
    if isinstance(value,dict):
        return {k:private_clean(v) for k,v in value.items() if k not in ex.FORBIDDEN_RELEASE_KEYS and k not in {'project','project_id','quota_project','quota_project_id','account','account_email'}}
    if isinstance(value,list):return [private_clean(v) for v in value]
    if isinstance(value,str):
        value=re.sub(r'projects/[^/\s]+/', 'projects/[REDACTED_PROJECT]/',value)
        for term in PRIVATE_TERMS:value=value.replace(term,'[REDACTED_PRIVATE_IDENTIFIER]')
    return value

def record_unit(unit,root):
    inputs=[]
    def read(path,lines=False):
        before=path.stat();raw=path.read_bytes();after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'source changed during export'
        inputs.append({'path':path.relative_to(unit).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)
    summary=read(unit/'unit_summary.json');terminal(summary)
    assert summary['model_id']==MODEL and summary['model_key']=='robotics' and summary['seed']==2026
    assert summary['goal']['source']=='canonical_world_gcond'
    for condition in ('full','neither','trace_status'):
        state=read(unit/'branches'/condition/'state.json')
        assert state['completed'] and len(state['attempts'])==summary['conditions'][condition]['attempt_count']
        if condition=='neither':assert len(state['attempts'])==8,'ER2 retry arm must retain all eight attempts'
    attempts=[];calls=[];sides=[]
    for p in walk(unit,{'attempt_record.json','provider_calls.jsonl'}):
        if 'transfer_sidecars' in p.relative_to(unit).parts:continue
        if p.name=='attempt_record.json':
            item=ex.compact_attempt_record(read(p));item['source']=p.relative_to(unit).as_posix();attempts.append(item)
        else:
            for item in read(p,lines=True):
                item['source']=p.relative_to(unit).as_posix();calls.append(item)
    side_summary=read(unit/'transfer_sidecars/summary.json');sidecar_complete(side_summary,summary['conditions'])
    sides.append({'source':'transfer_sidecars/summary.json','record':side_summary})
    for p in walk(unit/'transfer_sidecars',{'probe_selection.json','result.json','provider_calls.jsonl'}):
        values=read(p,lines=True) if p.name.endswith('.jsonl') else [read(p)]
        sides.extend({'source':p.relative_to(unit).as_posix(),'record':v} for v in values)
    record=private_clean({'schema_version':1,'source_unit':unit.relative_to(root/'episodes').as_posix(),
        'manifest':read(unit/'unit_manifest.json'),'summary':summary,'provider_calls':calls,
        'attempt_records':attempts,'excluded_unreadable_attempt_records':[],
        'sidecar_records':sides,'export_source_files':inputs,
        'release_provenance':{'model_family':'Gemini Robotics ER2','original_model_key':'robotics',
          'not_interchangeable_with_er16':True,'new_model_calls_during_export':0}})
    scan(record,unit.name)
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--icloud-helper',type=Path);p.add_argument('--private-redactions',type=Path,help='Private JSON array of literal identifiers to redact; never distribute this file.');args=p.parse_args()
    if args.private_redactions:PRIVATE_TERMS.extend(json.loads(args.private_redactions.read_text()))
    os.nice(10);args.cache.mkdir(parents=True,exist_ok=True)
    manifest=private_clean(ex.read_json(args.source_root/'canonical132-manifest.json'));assert manifest['model']['model_id']==MODEL
    units=sorted((args.source_root/'episodes/robotics/seed_2026').iterdir());assert len(units)==132 and all(u.is_dir() for u in units)
    out=args.repo/'model_outputs'/SNAPSHOT;out.mkdir(parents=True,exist_ok=True)
    def export_one(unit):
        cache=args.cache/(unit.name+'.jsonl.gz')
        if not cache.exists():
            record=record_unit(unit,args.source_root);tmp=cache.with_suffix('.tmp');ex.write_gzip_jsonl(tmp,[record]);tmp.replace(cache)
        with gzip.open(cache,'rt') as f:record=json.loads(f.readline())
        scan(record,unit.name);assert record['summary']['model_id']==MODEL
        return cache
    # Small batches keep iCloud hydration and disk use bounded; two readers.
    for start in range(0,len(units),8):
        batch=units[start:start+8]
        if args.icloud_helper:
            files=[];size=0
            for unit in batch:
                if (args.cache/(unit.name+'.jsonl.gz')).exists():continue
                for f in walk(unit,{'unit_summary.json','unit_manifest.json','state.json','attempt_record.json','provider_calls.jsonl','probe_selection.json','result.json','summary.json'}):
                    if f.stat().st_flags & 0x40000000 and size+f.stat().st_size<=64*2**20:
                        files.append({'path':str(f)});size+=f.stat().st_size
            request=args.cache/'prefetch.private.json';request.write_text(json.dumps(files));request.chmod(0o600)
            subprocess.run([str(args.icloud_helper),'download-batch',str(request)],stdout=subprocess.DEVNULL,check=True,timeout=30)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(export_one,batch))
        print(json.dumps({'er2_units_exported':min(start+8,132),'total':132}),flush=True)
    records=[]
    for unit in units:
        with gzip.open(args.cache/(unit.name+'.jsonl.gz'),'rt') as f:records.append(json.loads(f.readline()))
    shards=[]
    for index,start in enumerate(range(0,len(records),25)):
        path=out/'data/robotics_er2'/f'units-{index:04d}.jsonl.gz';rows=records[start:start+25];ex.write_gzip_jsonl(path,rows)
        shards.append({'path':path.relative_to(out).as_posix(),'unit_count':len(rows),'bytes':path.stat().st_size,'sha256':ex.sha256_file(path)})
    counts=[]
    for cond in ('full','neither','trace_status'):
        branches=[r['summary']['conditions'][cond] for r in records]
        counts.append({'model_id':MODEL,'condition':cond,'completed_episodes':132,'complete_sidecars':132,
                       'solved_episodes':sum(bool(b['solved_by_budget']) for b in branches),
                       'valid_attempts':sum(b['attempt_count'] for b in branches)})
    with (out/'coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(counts[0]),lineterminator='\n');w.writeheader();w.writerows(counts)
    released={'schema_version':1,'snapshot_id':SNAPSHOT,'model_id':MODEL,'seed':2026,'completed_main_episodes':132,'complete_canonical_sidecars':132,
              'benchmark_indices':[r['summary']['goal']['benchmark_index'] for r in records],'collection_manifest':manifest,'shards':shards,'coverage':counts,
              'scope':'Canonical extension; distinct from ER1.6 and from the original three-model broad goal bank.',
              'analysis_population_note':'Availability of a sidecar does not imply eligibility for every clean-probe analysis.',
              'excluded':'Credentials, account identifiers, absolute paths, goal-switch tasks, and regenerable images/full traces.',
              'model_calls_during_export':0}
    scan(released,'snapshot_manifest.json');(out/'snapshot_manifest.json').write_text(json.dumps(released,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'er2_export_complete':True,'episodes':132,'sidecars':132,'shards':len(shards)}),flush=True)

if __name__=='__main__':main()
