#!/usr/bin/env python3
"""Validate benchmark assets, finalized model records and release checksums."""
import argparse,gzip,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BAD={'BackUp_8','Balance_8','Falling_1'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify(root=ROOT,geometry=False):
 b=root/'benchmark';sums=json.loads((b/'checksums.json').read_text())
 for p,h in sums.items():assert sha(b/p)==h,p
 tasks=json.loads((b/'tasks.json').read_text())['goals'];idx=json.loads((b/'asset_index.json').read_text())['puzzles']
 assert len(tasks)==1692 and len({g['balanced_goal_id']for g in tasks})==1692 and len(idx)==132
 for key,e in idx.items():
  for p,h in [('world_path','world_sha256'),('screenshot_path','screenshot_sha256'),('tool_path','tool_sha256')]:assert sha(b/e[p])==e[h],(key,p)
 for g in tasks:
  e=idx[g['run_name']];assert g['environment_path']==e['world_path'] and g['initial_scene_path']==e['screenshot_path']and g['tool_path']==e['tool_path']
 battery=root/'data/vqa/battery'
 for p,h in json.loads((battery/'freeze_sha256.json').read_text()).items():assert sha(battery/p)==h,('VQA freeze',p)
 vqa=json.loads((root/'data/vqa/source_map.json').read_text());returned=set();corrected=0
 for entry in vqa:
  for field in ('raw_file','request_file'):
   if field not in entry:continue
   source=entry[field]
   if '::'in source:
    container,member=source.split('::',1)
    with zipfile.ZipFile(root/container)as z:raw=z.read(member)
   else:raw=(root/source).read_bytes()
   assert hashlib.sha256(raw).hexdigest()==entry[field+'_sha256']
  if 'raw_file'in entry:
   assert entry['call_id']not in returned;returned.add(entry['call_id'])
   corrected+=entry['origin']=='corrected_rerun'
 assert len(returned)==198 and corrected==9
 if geometry:subprocess.run([sys.executable,str(root/'scripts/release/verify_scene_geometry.py')],check=True)
 d=root/'data/finalized';manifest=json.loads((d/'manifest.json').read_text());lineage={(x['model'],x['goal']):x for x in manifest['lineage']};seen=set();changed={};counts={};probe_errors=set()
 from validate_records import terminal,scan
 expected_ids={'gpt5':'openai/gpt-5','gemini31pro':'google/gemini-3.1-pro-preview','qwen36plus':'qwen/qwen3.6-plus','robotics_er2':'gemini-robotics-er-2-preview'}
 for shard in manifest['shards']:
  p=d/shard['path'];assert sha(p)==shard['sha256'];n=0;m=shard['model']
  with gzip.open(p,'rt')as f:
   for line in f:
    r=json.loads(line);g=r['summary']['goal'];pair=(m,g['balanced_goal_id']);assert pair not in seen;seen.add(pair);n+=1;counts[m]=counts.get(m,0)+1
    terminal(r['summary']);scan(r,shard['path']);assert r['summary']['model_id']==expected_ids[m]
    for entry in r['sidecar_records']:
     q=entry['record']
     if entry['source'].endswith('/result.json')and not q.get('schema_valid')and(q.get('last_provider_metadata')or{}).get('finish_reason')=='error':probe_errors.add((m,g['balanced_goal_id'],entry['source']))
    e=idx[g['puzzle_key']];assert e['screenshot_sha256']==g['asset_screenshot_sha256']and e['world_sha256']==g['asset_world_sha256']
    l=lineage[pair];assert canonical(r)==l['final_record_sha256']
    if l['origin']=='preserved_original':assert l['original_record_sha256']==l['final_record_sha256']and g['puzzle_key'].rsplit('_',1)[0]not in BAD
    else:assert g['puzzle_key'].rsplit('_',1)[0]in BAD;changed[m]=changed.get(m,0)+1
  assert n==shard['records']
 assert seen==set(lineage)
 assert probe_errors=={(p['model'],p['goal'],p['query'])for p in manifest.get('missing_probes',[])}
 report={'passed':True,'benchmark_goals':len(tasks),'layout_direction_cells':len(idx),'vqa_returned_responses':len(returned),'vqa_corrected_calls':corrected,'completed_model_units':counts,'corrected_units':changed,'missing_replacements':manifest['missing_replacements'],'missing_probes':manifest.get('missing_probes',[])}
 print(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry',action='store_true');a=p.parse_args();verify(geometry=a.geometry)
