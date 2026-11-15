#!/usr/bin/env python3
"""Build a versioned model release by replacing only three corrected layouts.

Historical archives are read-only. A missing replacement is never substituted
with the inconsistent historical observation or counted as a task failure.
"""
import argparse,gzip,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT/'archive/pre_correction_2026-11-14'
sys.path.insert(0,str(LEGACY/'scripts/release'))
import export_model_outputs as export
from validate_documented_snapshot import terminal,sidecar_complete,scan,fingerprint
MODELS=('gpt5','gemini31pro','qwen36plus','robotics_er2')
LAYOUTS={'BackUp_8','Balance_8','Falling_1'}
def records(model):
 snap='robotics_er2_canonical132_seed2026_2026-11-08' if model=='robotics_er2' else 'snapshot_seed2026_documented_gaps_2026-11-05'
 for path in sorted((LEGACY/'model_outputs'/snap/'data'/model).glob('*.jsonl.gz')):
  with gzip.open(path,'rt')as f:
   for line in f:yield json.loads(line)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'data/finalized');p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
 if a.output.exists()and any(a.output.iterdir()):raise ValueError('Output must be new/empty')
 work=json.loads((a.run_root/'preflight/worklist.json').read_text());expected={(w['model'],w['goal_id']):i for i,w in enumerate(work)};ready={};missing=[];missing_probes=[]
 for pair,i in expected.items():
  f=a.run_root/'status'/f'{i:03d}.json';s=json.loads(f.read_text())if f.exists()else {}
  if s.get('phase')!='complete':
   if s.get('phase')=='needs_attention'and s.get('probe_gaps'):
    unit=Path(s['unit_dir']);note=json.loads((unit/'probe_provider_error_recovery.json').read_text())
    assert note.get('finished_unix')and note['main_records_unchanged']and note['exit_code']==0
    assert sorted(note['unresolved_queries'])==sorted(s['probe_gaps'])
    assert export.sha256_file(unit/'unit_summary.json')==note['main_summary_sha256']
    terminal(json.loads((unit/'unit_summary.json').read_text()))
    for rel in s['probe_gaps']:
     q=json.loads((unit/rel).read_text());assert not q.get('schema_valid')and q['last_provider_metadata']['finish_reason']=='error'
     missing_probes.append({'model':pair[0],'goal':pair[1],'query':rel,'reason':'Provider error after one reviewed restart; excluded as missing, not scored zero.'})
   else:missing.append({'model':pair[0],'goal':pair[1],'phase':s.get('phase','missing')});continue
  ready[pair]=Path(s['unit_dir'])
 if missing and not a.allow_incomplete:raise ValueError(f'{len(missing)} replacement units not complete; no dataset written')
 idx=json.loads((ROOT/'benchmark/asset_index.json').read_text())['puzzles'];a.output.mkdir(parents=True,exist_ok=True);manifest={'version':'corrected-scenes-v1','selection_seed':2026,'replaced_layouts':sorted(LAYOUTS),'source_commit':'4748d2790eb325513829b6296a6d957d428c2903','missing_replacements':missing,'missing_probes':missing_probes,'models':{},'lineage':[],'shards':[]}
 for model in MODELS:
  final=[];changed=0;preserved=0
  for old in records(model):
   g=old['summary']['goal'];gid=g['balanced_goal_id'];layout=g['puzzle_key'].rsplit('_',1)[0];oldhash=fingerprint(old)
   if layout in LAYOUTS:
    unit=ready.get((model,gid))
    if unit is None:continue
    new=export.unit_record(unit/'unit_summary.json',a.run_root/'episodes');changed+=1
    for filename,field in [('token_budget_recovery_calls.jsonl','token_budget_recovery_calls'),('token_recovery_raw.jsonl','token_recovery_raw_responses')]:
     if (unit/filename).exists():new[field]=export.sanitize([json.loads(line)for line in (unit/filename).read_text().splitlines()])
    for backup in sorted(unit.glob('transfer_sidecars/**/*.provider_error_before_recovery.json')):new['sidecar_records'].append({'source':str(backup.relative_to(unit)),'record':export.sanitize(json.loads(backup.read_text()))})
    if (unit/'probe_provider_error_recovery.json').exists():new['probe_provider_error_recovery']=export.sanitize(json.loads((unit/'probe_provider_error_recovery.json').read_text()))
    new['collection_provenance']={'release':'corrected-scenes-v1','input_correction':'2026-11-14','collection_started':'2026-11-15','historical_record_sha256':oldhash,'probe_policy':'Original fixed placements; newly simulated endpoint truth; no outcome-based replacement selection.'}
    assert new['summary']['goal']['balanced_goal_id']==gid
    for key in ('signature','asset_world_sha256','attempt_budget'):assert new['summary']['goal'][key]==g[key],(gid,key)
    bundles=[z['record']for z in new['sidecar_records']if z['source']=='transfer_sidecars/summary.json']
    w=work[expected[(model,gid)]]
    if w['selection']:
     assert len(bundles)==1;sidecar_complete(bundles[0],new['summary']['conditions'])
     for pn in ('early_probe','terminal_probe'):assert bundles[0]['probe_selection'][pn]['coords']==w['selection'][pn]['coords']
   else:new=old;preserved+=1;assert fingerprint(new)==oldhash
   terminal(new['summary']);scan(new,model+'/'+gid)
   ng=new['summary']['goal'];asset=idx[ng['puzzle_key']]
   assert ng['asset_screenshot_sha256']==asset['screenshot_sha256'] and ng['asset_world_sha256']==asset['world_sha256']
   manifest['lineage'].append({'model':model,'goal':gid,'origin':'corrected_rerun'if layout in LAYOUTS else 'preserved_original','original_record_sha256':oldhash,'final_record_sha256':fingerprint(new)})
   final.append(new)
  final.sort(key=lambda r:r['summary']['goal']['balanced_goal_id'])
  for start in range(0,len(final),100):
   path=a.output/model/f'units-{start//100:04d}.jsonl.gz';batch=final[start:start+100];export.write_gzip_jsonl(path,batch)
   manifest['shards'].append({'model':model,'path':str(path.relative_to(a.output)),'records':len(batch),'sha256':export.sha256_file(path)})
  manifest['models'][model]={'completed_units':len(final),'replacements':changed,'preserved':preserved};print(model,manifest['models'][model],flush=True)
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (a.output/'README.md').write_text('''# Finalized model observations

`manifest.json` identifies the authoritative shards, SHA-256 checksums, completed units, missing replacements, and record-level replacement lineage. Read each `*.jsonl.gz` shard with Python's `gzip.open(path, "rt")`; each line is one model-task episode across its feedback conditions.

- `summary`: model, seed, task definition, branch success and valid-attempt counts.
- `attempt_records`: executed placements, goal outcomes and simulated object endpoints.
- `provider_calls`: prompts, image hashes, raw responses and normalized response payloads.
- `sidecar_records`: fixed early/terminal novel interventions, predictions and scoring truth.
- `collection_provenance` on replacements: correction date and original-record fingerprint.

Only BackUp_8, Balance_8 and Falling_1 are replaced. Unaffected observations retain their original fingerprints; historical source files remain byte-for-byte in the archive. Interrupted responses and recovery calls are retained rather than converted into failures.

Completed model-unit counts are not interchangeable with analysis sample sizes. Use `results/populations.json` for the exact main, broad-control and matched-probe memberships, and `results/method.json` for estimators. Coordinates use the original 600-pixel Cartesian scene convention. Human participant-level records are not included.
''')
if __name__=='__main__':main()
