"""Rescore frozen deterministic replay caches with the released replay functions."""
import collections,csv,hashlib,importlib.util,json,math
from pathlib import Path
import argparse,zipfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--output-dir',type=Path,required=True)
a=parser.parse_args();R=a.repo_root.resolve();O=a.output_dir.resolve()
if O.exists() and any(O.iterdir()):raise ValueError('Use a new/empty output directory')
(O/'results').mkdir(parents=True,exist_ok=True);C=O/'cache'
source=R/'model_outputs/coordinate_replay_snapshot_2026-07'
expected=json.loads((source/'source_sha256.json').read_text())
with zipfile.ZipFile(source/'frozen_replay_cache.zip') as archive:
 if set(archive.namelist())!=set(expected):raise ValueError('Unexpected archive members')
 for member in archive.namelist():
  p=Path(member)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
 archive.extractall(C)
for name,h in expected.items():
 if hashlib.sha256((C/name).read_bytes()).hexdigest()!=h:raise ValueError('Cache checksum mismatch')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
runner=load('original_coordinate_runner',R/'scripts/evaluation/run_coordinate_replay.py');analysis=load('original_coordinate_analysis',R/'scripts/evaluation/analyze_coordinate_replay.py')
manifest=json.loads((R/'task_configs/benchmark_1692_seed2026.json').read_text());goals={g['balanced_goal_id']:g for g in manifest['goals']}
actions=list(runner.read_jsonl(C/'accepted_actions.jsonl'));cells={p.stem:json.loads(p.read_text()) for p in (C/'cells').glob('*.json')};taskrows=list(runner.read_jsonl(C/'replay_tasks.jsonl'));tasks={x['puzzle_key']:x for x in taskrows}
for key,cell in cells.items():
 assert cell['input_sha256']==tasks[key]['input_sha256']
 assert cell['rounding_grid_px']==10 and tasks[key]['jitter_offsets']==[list(x) for x in runner.JITTER_OFFSETS]
 for coord,rounded in cell['rounded_by_original'].items():assert rounded==list(runner.round_point_to_grid([int(x) for x in coord.split(',')]))
grouped=collections.defaultdict(list);sequences=collections.defaultdict(list);issues=[];skipped=collections.Counter();newrows=[]
for a in actions:
 if a['model_key'] not in ['gpt5','gemini31pro','qwen36plus']:skipped[a['model_key']]+=1;continue
 key=a['puzzle_key'];cell=cells.get(key);goal=goals.get(a['goal_id'])
 if cell is None or goal is None:issues.append({'puzzle':key,'missing':'cell' if cell is None else 'goal'});continue
 xy=a['original_coords'];rounded=cell['rounded_by_original'][runner.coordinate_key(xy)];sim=cell['simulations'];orig=sim[runner.coordinate_key(xy)];rnd=sim[runner.coordinate_key(rounded)];orig_s=runner.goal_success_from_simulation(goal,orig);rnd_s=runner.goal_success_from_simulation(goal,rnd)
 jitter=[];valid=[]
 if a['jitter_scope']:
  for dx,dy in runner.JITTER_OFFSETS:
   v=sim[runner.coordinate_key([xy[0]+dx,xy[1]+dy])];valid.append(bool(v['valid']));jitter.append(runner.goal_success_from_simulation(goal,v))
 row={'model':a['model_key'],'condition':a['condition'],'goal':a['goal_id'],'layout':key.rsplit('_',1)[0],'gravity':a['gravity'],'attempt':a['attempt'],'seed':a['seed'],'original':orig_s,'logged_original':bool(a['original_success']),'original_valid':bool(orig['valid']),'rounded':rnd_s,'rounded_valid':bool(rnd['valid']),'rounding_eligible':bool(orig['valid'] and rnd['valid']),'jitter_scope':bool(a['jitter_scope']),'jitter_agree_count':sum(v==orig_s for v in jitter),'jitter_valid_count':sum(valid),'jitter_count':len(jitter),'isolated':bool(orig_s and jitter and not any(jitter)),'displacement':math.dist(xy,rounded)}
 newrows.append(row);grouped[(a['model_key'],a['condition'])].append(row);sequences[(a['model_key'],a['seed'],a['goal_id'],a['condition'])].append(row)
assert not issues,issues[:2]
seqrows=[]
for (model,seed,gid,c),rr in sequences.items():
 rr=sorted(rr,key=lambda r:r['attempt']);assert [r['attempt'] for r in rr]==list(range(1,len(rr)+1))
 orig=any(r['original'] for r in rr);rounded=any(r['rounded'] and r['rounding_eligible'] for r in rr);allvalid=all(r['rounding_eligible'] for r in rr)
 seqrows.append({'model_key':model,'seed':seed,'goal_id':gid,'condition':c,'gravity':rr[0]['gravity'],'layout':rr[0]['layout'],'attempts':len(rr),'original_solve_by_8':str(orig),'rounded_solve_by_8_invalid_as_failure':str(rounded),'all_actions_rounding_eligible':str(allvalid),'rounded_solve_by_8_when_all_valid':str(rounded) if allvalid else '', 'solve_by_8_agreement_invalid_as_failure':str(orig==rounded)})
summaries=[]
for (m,c),rr in sorted(grouped.items()):
 ss=[x for x in seqrows if x['model_key']==m and x['condition']==c];j=[x for x in rr if x['jitter_scope']];sj=[x for x in j if x['original']];eligible=[x for x in rr if x['rounding_eligible']]
 summaries.append({'model':m,'condition':c,'seed':2026,'cells':len({(x['layout'],x['gravity']) for x in rr}),'episodes':len(ss),'actions':len(rr),'episode_agree_n':sum(x['solve_by_8_agreement_invalid_as_failure']=='True' for x in ss),'episode_agreement':sum(x['solve_by_8_agreement_invalid_as_failure']=='True' for x in ss)/len(ss),'original_solved_n':sum(x['original_solve_by_8']=='True' for x in ss),'rounded_solved_n':sum(x['rounded_solve_by_8_invalid_as_failure']=='True' for x in ss),'rounding_eligible_actions':len(eligible),'invalid_rounded_actions':sum(not x['rounded_valid'] for x in rr),'eligible_action_agreement':sum(x['original']==x['rounded'] for x in eligible)/len(eligible),'jitter_actions':len(j),'neighbor_replays':sum(x['jitter_count'] for x in j),'neighbor_agree_n':sum(x['jitter_agree_count'] for x in j),'neighbor_agreement':sum(x['jitter_agree_count'] for x in j)/sum(x['jitter_count'] for x in j),'invalid_neighbors':sum(x['jitter_count']-x['jitter_valid_count'] for x in j),'successful_jitter_actions':len(sj),'isolated_successes':sum(x['isolated'] for x in sj),'isolated_success_fraction':sum(x['isolated'] for x in sj)/len(sj),'logged_vs_replayed_outcome_mismatches':sum(x['original']!=x['logged_original'] for x in rr),'max_rounding_displacement':max(x['displacement'] for x in rr)})
(O/'results/coordinate').mkdir(exist_ok=True)
for filename,rows in [('coordinate_summary.csv',summaries),('sequence_reconciliation.csv',seqrows),('action_reconciliation.csv',newrows),('original_analysis_sequence_summary.csv',analysis.sequence_summary(seqrows))]:
 with (O/'results/coordinate'/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'cache_cells':len(cells),'recorded_actions_total':len(actions),'main_model_actions':len(newrows),'excluded_other_model_actions':dict(skipped),'main_model_episodes':len(seqrows),'jitter_scope_verified':all(r['jitter_scope']==(r['attempt']==1 or r['logged_original']) for r in newrows),'seed_values':sorted({r['seed'] for r in newrows}),'mechanical_rounding_all_cached_points_verified':True,'no_projection_or_best_neighbor_selection':True,'missing_caches_or_goals':issues,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [C/'accepted_actions.jsonl',C/'replay_tasks.jsonl',R/'scripts/evaluation/run_coordinate_replay.py',R/'scripts/evaluation/analyze_coordinate_replay.py']},'all_condition_ranges':{k:[min(r[k] for r in summaries),max(r[k] for r in summaries)] for k in ['episode_agreement','neighbor_agreement','isolated_success_fraction']},'three_main_arm_ranges':{k:[min(r[k] for r in summaries if r['condition'] in ['full','neither','trace_status']),max(r[k] for r in summaries if r['condition'] in ['full','neither','trace_status'])] for k in ['episode_agreement','neighbor_agreement','isolated_success_fraction']},'summary':summaries}
(O/'results/coordinate/audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
