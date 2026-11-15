"""Resume one corrected unit with its collection-specific frozen protocol."""
import argparse,importlib.util,json,sys,hashlib,os,time,gzip,math,copy
from pathlib import Path
O=Path(__file__).resolve().parent
R=Path('<HISTORICAL_REPOSITORY>')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--mock',action='store_true');p.add_argument('--cap-usd',type=float,default=10);p.add_argument('--resume-error',action='store_true',help='Resume only after the saved interruption has been individually reviewed.');cli=p.parse_args()
w=json.loads((O/'preflight/worklist.json').read_text())[cli.index];key=w['model'];is_er=key=='robotics_er2';root=O/'er2_runtime' if is_er else R
if cli.execute:
 import fcntl
 (O/'locks').mkdir(exist_ok=True);unit_lock=(O/'locks'/f'{cli.index:03d}.lock').open('a')
 try:fcntl.flock(unit_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print('Unit is already running in another queue; no duplicate calls made');sys.exit(0)
 previous_status=O/'status'/f'{cli.index:03d}.json'
 if previous_status.exists()and json.loads(previous_status.read_text()).get('phase')=='complete':
  print('Unit already complete; no calls made');sys.exit(0)
 if previous_status.exists()and json.loads(previous_status.read_text()).get('phase')=='needs_attention'and not cli.resume_error:
  print('Saved interruption requires review; no automatic restart or calls made');sys.exit(2)
r=load('corrected_forked',root/'scripts/experiment/forked_feedback_runner.py');side=load('corrected_sidecar',root/'scripts/experiment/transfer_sidecar_runner.py')
if is_er:r.MODEL_SPECS[key]=r.ModelSpec(key,'gemini-robotics-er-2-preview',.2,'medium','Gemini Robotics-ER 2 Preview')
argv=sys.argv;sys.argv=['rerun','--provider','mock','--model',key,'--seed','2026','--conditions',*w['conditions'],'--max-tokens','8192','--fixed-budget','8','--retain-truth-traces','--output-root',str(O/('mock_episodes' if cli.mock else 'episodes')),'--asset-index',str(R/'task_configs/asset_index_132_scene_corrected_2026-11-14.json')]
a=r.parse_args();sys.argv=argv
assets=r.load_or_build_asset_index(r.load_json(a.manifest)['goals'],asset_index_path=a.asset_index,asset_roots=(R,),rebuild=False)
goals=r.load_balanced_goals(a.manifest,asset_index=assets,budget_mode='fixed',fixed_budget=8)
g=next(g for g in goals if g['balanced_goal_id']==w['goal_id']);assert g['puzzle_key'].rsplit('_',1)[0] in {'BackUp_8','Balance_8','Falling_1'}
bank=r.base.GoalBank(root/'task_configs',a.goal_builder)
old=json.load(gzip.open(O/w['historical_source'],'rt'))
# Before any inference, replay the original first action on the unchanged world.
old_first=old['summary']['shared_attempt_1'];coords=old_first['coords']
ctx=bank.context_for(g);targets=r.prediction_targets_for_goal(g,bank)
checkdir=O/'preflight/simulator_checks'/key/g['balanced_goal_id'];checkdir.mkdir(parents=True,exist_ok=True)
# Native trial path applies the same canonical dwell and final-state predicates.
kwargs=dict(goal=g,goal_bank=bank,coords=coords,attempt_number=1,attempt_budget=8,attempt_dir=checkdir,world_path=Path(g['environment_json']),frames_condition=False,trace_condition=False,stop_on_success=True,frame_count=32,trace_state_count=32)
replayed=r.simulate_attempt(**kwargs)
assert replayed['goal_succeeded']==old_first['goal_succeeded'], 'Historical outcome replay mismatch'
reference=next(z for z in old['attempt_records'] if z['source'].startswith('shared_attempt_1/') and z['coords']==coords)
original_objects=reference['truth']['prediction_endpoints']['objects']
replay_deltas=[]
for obj,truth in replayed['truth']['prediction_endpoints']['objects'].items():
 replay_deltas.append({'object':obj,'endpoint_difference_px':math.dist([truth['final_x'],truth['final_y']],[original_objects[obj]['final_x'],original_objects[obj]['final_y']])})
selection=copy.deepcopy(w['selection'])
if w['selection'] is not None:
 for probe_name in ('early_probe','terminal_probe'):
  probe=w['selection'][probe_name];xy=probe['coords'];td=checkdir/probe_name
  result=bank.evaluate_goal_at(g,xy,trace_dir=td);assert result['valid']
  truth=side.endpoint_truth(r.load_json(td/f'trace_{xy[0]}_{xy[1]}.json'))
  for obj,expected in probe['endpoints'].items():
   assert truth[obj]['state']==expected['state'] and truth[obj].get('exit_side')==expected.get('exit_side')
   if expected['state']=='in_scene':replay_deltas.append({'probe':probe_name,'object':obj,'endpoint_difference_px':math.dist(truth[obj]['point'],expected['point'])})
  # Preserve the exact intervention while recording truth from this runtime.
  trace_path=td/f'trace_{xy[0]}_{xy[1]}.json';trace=r.load_json(trace_path)
  selection[probe_name].update(endpoints=truth,features=side.rollout_features(trace,xy),trace_sha256=r.sha256_file(trace_path))
r.atomic_write_json(checkdir/'replay_comparison.json',{'historical_success':old_first['goal_succeeded'],'replayed_success':replayed['goal_succeeded'],'endpoint_differences':replay_deltas,'policy':'Same physics code and parameters; local floating-point replay is not guaranteed bit-identical to historical workers. New records retain newly simulated truth; original records remain unchanged.'})
print(json.dumps({'unit':cli.index,'goal':g['balanced_goal_id'],'model':key,'screenshot_sha256':r.sha256_file(Path(g['initial_screenshot'])),'world_sha256':r.sha256_file(Path(g['environment_json'])),'targets':targets,'execute':cli.execute,'mock':cli.mock}),flush=True)
if not cli.execute and not cli.mock:
 print('preflight: runner loaded, goal and corrected assets resolved');sys.exit(0)
if cli.mock:provider=r.make_provider(a,r.MODEL_SPECS[key])
elif is_er:
 from providers import DirectProvider
 provider=DirectProvider('robotics',r.build_system_prompt('full'),O/'providers/robotics_er2',cli.cap_usd)
else:
 cls=r.OpenRouterProvider
 if key=='qwen36plus':
  from qwen_recovery import provider_class
  cls=provider_class(cls,r.goal_unit_dir(a.output_root,key,2026,g))
 provider=cls(api_key=r.load_credential_file(Path.home()/'.config/openrouter/api_key','OPENROUTER_API_KEY'),model_spec=r.MODEL_SPECS[key],seed=2026,system_prompt=r.build_system_prompt('full'),base_url='https://openrouter.ai/api/v1',max_tokens=8192,request_timeout_s=900 if key=='qwen36plus'else 300,max_provider_cost_usd=cli.cap_usd)
if not cli.mock:
 from budget_guard import Guard
 if is_er:
  ledger=O/'providers/robotics_er2/provider-usage.jsonl'
  baseline=sum(json.loads(l)['estimated_cost_usd']for l in ledger.read_text().splitlines())if ledger.exists()else 0
  provider=Guard(provider,O/'budget','robotics',10,(provider.input_price,provider.output_price),baseline)
 else:
  prices={'gpt5':(1.25,10),'gemini31pro':(2,12),'qwen36plus':(.325,1.95)}
  provider=Guard(provider,O/'budget','openrouter',188,prices[key])
status=O/'status'/f'{cli.index:03d}.json';status.parent.mkdir(exist_ok=True)
def save(phase,**extra):r.atomic_write_json(status,{'index':cli.index,'model':key,'goal_id':g['balanced_goal_id'],'phase':phase,'updated_unix':time.time(),**extra})
save('main_running')
try:
 provider.current_unit=g['balanced_goal_id'];provider.current_phase='main'
 summary=r.run_unit(args=a,provider=provider,model_spec=r.MODEL_SPECS[key],goal=g,goal_bank=bank,output_root=a.output_root)
 unit=r.goal_unit_dir(a.output_root,key,2026,g)
 for condition in w['conditions']:
  st=r.load_json(unit/'branches'/condition/'state.json');attempts=st.get('attempts',[])
  hidden=not r.CONDITIONS[condition]['status'];absorbed=(not is_er and bool(summary['shared_attempt_1']['goal_succeeded']))
  terminal=len(attempts)==8 or ((not hidden or absorbed) and any(z['goal_succeeded'] for z in attempts))
  if not st.get('completed') or not terminal:raise RuntimeError('Branch not terminal: '+condition)
 if w['selection'] is not None:
  save('probes_running');provider.current_phase='heldout_predictions'
  selpath=unit/'transfer_sidecars/probe_selection.json'
  if not selpath.exists():r.atomic_write_json(selpath,selection)
  else:
   for probe in ('early_probe','terminal_probe'):
    assert r.load_json(selpath)[probe]['coords']==selection[probe]['coords']
  # Selection is frozen from the historical collection, not regenerated using new actions.
  side.prepare_probe_bank=lambda **kw:None
  result=side.run_unit_sidecars(args=a,provider=provider,goal=g,goal_bank=bank,unit_dir=unit,probe_cache=O/'unused_frozen_probe_cache')
  if not side.sidecar_summary_complete(result,expected_conditions=w['conditions']):raise RuntimeError('Probe bundle not terminal')
 save('complete',unit_dir=str(unit),cost_usd=provider.cumulative_cost_usd)
except Exception as e:
 save('needs_attention',error=f'{type(e).__name__}: {e}');raise
