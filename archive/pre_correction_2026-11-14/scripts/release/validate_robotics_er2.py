#!/usr/bin/env python3
"""Validate the separate ER2 archive, exact task matching, and public contents."""
import argparse,csv,gzip,json,re
from pathlib import Path
import export_model_outputs as ex
from validate_documented_snapshot import terminal,sidecar_complete,scan
from export_robotics_er2 import MODEL,SNAPSHOT

def require(value,message):
    if not value:raise ValueError(message)

def validate(repo):
    root=repo/'model_outputs'/SNAPSHOT
    manifest=json.loads((root/'snapshot_manifest.json').read_text())
    require(manifest['model_id']==MODEL and manifest['seed']==2026,'ER2 model identity or seed mismatch')
    require(manifest['completed_main_episodes']==manifest['complete_canonical_sidecars']==132,'ER2 coverage mismatch')
    reference={}
    prior=repo/'model_outputs/snapshot_seed2026_documented_gaps_2026-11-05'
    for path in sorted((prior/'data/gpt5').glob('*.jsonl.gz')):
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line);g=r['summary']['goal']
                if g['source']=='canonical_world_gcond':reference[g['benchmark_index']]=g
    require(len(reference)==132,'Canonical comparison population missing')
    seen=set();paths=set();attempts={c:0 for c in ('full','neither','trace_status')};solves={c:0 for c in attempts};inputs=0
    for shard in manifest['shards']:
        path=root/shard['path'];require(path.resolve().is_relative_to(root.resolve()) and path not in paths,'Unsafe or duplicate shard')
        paths.add(path);require(path.stat().st_size==shard['bytes'] and ex.sha256_file(path)==shard['sha256'],'ER2 shard checksum mismatch')
        n=0
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line);scan(r,shard['path']);s=r['summary'];terminal(s)
                require(s['model_id']==MODEL and s['model_key']=='robotics' and s['seed']==2026,'Record model identity mismatch')
                require(set(s['conditions'])==set(attempts),'ER2 condition mismatch')
                g=s['goal'];index=g['benchmark_index'];require(index in reference and index not in seen,'Duplicate or unexpected ER2 task');seen.add(index)
                require(r['source_unit'].startswith('robotics/seed_2026/'),'ER2 source namespace mismatch')
                for key in ['balanced_goal_id','signature','asset_world_sha256','asset_screenshot_sha256','attempt_budget']:
                    require(g[key]==reference[index][key],'ER2 canonical task/asset mismatch: '+key)
                require(s['conditions']['neither']['attempt_count']==8,'ER2 retry arm stopped before eight attempts')
                require(not r['excluded_unreadable_attempt_records'] and r['attempt_records'] and r['provider_calls'],'Missing action records or raw responses')
                bundles=[v['record'] for v in r['sidecar_records'] if v['source']=='transfer_sidecars/summary.json']
                require(len(bundles)==1,'Missing ER2 sidecar summary');sidecar_complete(bundles[0],s['conditions'])
                for c,b in s['conditions'].items():attempts[c]+=b['attempt_count'];solves[c]+=bool(b['solved_by_budget'])
                require(r['export_source_files'],'Source hash lineage missing')
                for source in r['export_source_files']:
                    require(not Path(source['path']).is_absolute() and '..' not in Path(source['path']).parts,'Unsafe source label')
                    require(re.fullmatch('[0-9a-f]{64}',source['sha256']) and source['bytes']>=0,'Invalid source hash');inputs+=1
                n+=1
        require(n==shard['unit_count'],'ER2 shard record count mismatch')
    require(seen==set(reference)==set(manifest['benchmark_indices']),'ER2 canonical coverage mismatch')
    require(paths==set((root/'data').rglob('*.jsonl.gz')),'Unlisted ER2 shard')
    with (root/'coverage.csv').open() as f:coverage=list(csv.DictReader(f))
    require(len(coverage)==3,'ER2 coverage rows missing')
    for row,embedded in zip(coverage,manifest['coverage']):
        require(all(row[k]==str(v) for k,v in embedded.items()),'ER2 coverage CSV/manifest mismatch')
        c=row['condition'];require(int(row['solved_episodes'])==solves[c] and int(row['valid_attempts'])==attempts[c],'ER2 outcome counts mismatch')
    for path in root.rglob('*'):
        if path.is_file() and path.suffix not in ['.gz','.py']:
            scan(path.read_text(),path.relative_to(root).as_posix())
    return {'validation':'passed','model_id':MODEL,'completed_main':len(seen),'complete_sidecars':132,'shards':len(paths),'source_files_hashed':inputs,'solved_episodes':solves,'valid_attempts':attempts,'canonical_assets_match_gpt5':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);args=p.parse_args();print(json.dumps(validate(args.repo.resolve()),indent=2,sort_keys=True))
