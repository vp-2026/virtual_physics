#!/usr/bin/env python3
"""Trace the requested layouts and the additional Falling_1 finding."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

LAYOUTS={'BackUp_8','Balance_8','Falling_1'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();r=a.repo_root.resolve();out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new/empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    assets=json.loads((r/'task_configs/asset_index_132.json').read_text())['puzzles']
    rows=[];evidence=[];hashes={};structure=[];json_checked=0
    worlds={k:json.loads((r/'task_configs'/v['world_path']).read_text())['world']['objects']
            for k,v in assets.items() if k.rsplit('_',1)[0] in LAYOUTS}
    for snapshot in sorted((r/'model_outputs').iterdir()):
        if not (snapshot/'data').is_dir():continue
        sm=json.loads((snapshot/'snapshot_manifest.json').read_text())
        expected_shards={x['path']:x['sha256'] for x in sm['shards']}
        count=0
        for path in sorted((snapshot/'data').glob('*/*.jsonl.gz')):
            hh=hashlib.sha256(path.read_bytes()).hexdigest()
            assert hh==expected_shards[str(path.relative_to(snapshot))],('Shard checksum',str(path))
            # These are public model-only shards; no private or participant records.
            with gzip.open(path,'rt') as f:
                for line in f:
                    d=json.loads(line);summary=d.get('summary')or{};g=summary.get('goal')or{}
                    puzzle=g.get('puzzle_key','');layout=puzzle.rsplit('_',1)[0]
                    if layout not in LAYOUTS:continue
                    expected=assets[puzzle];calls=d.get('provider_calls',[]);count+=1
                    hashes[str(path.relative_to(r))]=hh
                    assert g.get('asset_world_sha256')==expected['world_sha256'],('World provenance',puzzle)
                    assert g.get('asset_screenshot_sha256')==expected['screenshot_sha256'],('Image provenance',puzzle)
                    image_calls=[c for c in calls if expected['screenshot_sha256'] in c.get('image_sha256',[])]
                    trace=[c for c in calls if c.get('source','').startswith('branches/trace_status/')]
                    assert not any(c.get('image_sha256') for c in trace), 'Unexpected image in JSON arm'
                    trace_initial=[c for c in trace if c.get('call_label')=='shared_attempt_1']
                    trace_geometry=[]
                    for c in trace_initial:
                        marker='Initial observable scene JSON:\n'
                        if marker not in c.get('prompt',''):continue
                        try:scene,_=json.JSONDecoder().raw_decode(c['prompt'].split(marker,1)[1])
                        except json.JSONDecodeError:continue
                        for obj in scene['visible_objects']:
                            raw=worlds[puzzle][obj['id']]
                            assert all(raw[k]==v for k,v in obj['geometry'].items()),('JSON scene geometry',puzzle,obj['id'])
                        trace_geometry.append(scene);json_checked+=1
                    row=dict(snapshot=snapshot.name,model=path.parent.name,goal_id=g.get('balanced_goal_id'),
                        puzzle_key=puzzle,layout=layout,seed=summary.get('seed'),
                        conditions=';'.join(sorted(summary.get('conditions',{}))),
                        screenshot_sha256=g.get('asset_screenshot_sha256'),world_sha256=g.get('asset_world_sha256'),
                        image_request_count=len(image_calls),json_request_count=len(trace),
                        screenshot_matches_release=g.get('asset_screenshot_sha256')==expected['screenshot_sha256'],
                        world_matches_release=g.get('asset_world_sha256')==expected['world_sha256'],
                        shard=str(path.relative_to(r)),source_unit=d.get('source_unit'))
                    rows.append(row)
                    if image_calls or trace_geometry:
                        evidence.append(dict(snapshot=snapshot.name,model=path.parent.name,goal_id=row['goal_id'],
                            matching_visual_requests=[dict(source=c['source'],label=c['call_label'],
                                timestamp_unix=c.get('timestamp_unix'),prompt_sha256=c.get('prompt_sha256'),
                                initial_image_sha256=c['image_sha256'][0]) for c in image_calls],
                            json_initial_scenes=trace_geometry))
        if count:print(snapshot.name,count,'affected archived units',flush=True)
    (out/'request_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (out/'shard_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with (out/'affected_archived_units.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    for name in ['paper_goals_1560.json','benchmark_1692_seed2026.json']:
        d=json.loads((r/'task_configs'/name).read_text());gg=[g for g in d['goals'] if g['run_name'].rsplit('_',1)[0] in LAYOUTS]
        structure.append(dict(manifest=name,total=len(d['goals']),affected=len(gg),
            by_puzzle=dict(collections.Counter(g['run_name']for g in gg)),
            affected_goal_ids=[g['balanced_goal_id']for g in gg]))
    (out/'affected_task_manifests.json').write_text(json.dumps(structure,indent=2)+'\n')
    summary=[]
    for (snap,model),rr in __import__('itertools').groupby(sorted(rows,key=lambda x:(x['snapshot'],x['model'])),key=lambda x:(x['snapshot'],x['model'])):
        rr=list(rr);summary.append(dict(snapshot=snap,model=model,n=len(rr),
            by_puzzle=dict(collections.Counter(x['puzzle_key']for x in rr)),
            visual_request_hash_matches=sum(x['image_request_count']for x in rr),
            world_hash_mismatches=sum(not x['world_matches_release']for x in rr)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(
        shard_manifest_checks_passed=True,matched_historical_world_and_image_hashes=True,
        json_initial_scenes_compared_to_world=json_checked,
        json_arm_image_requests=0,affected_archived_units=len(rows)),indent=2)+'\n')


if __name__=='__main__':main()
