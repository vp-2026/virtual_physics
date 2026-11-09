"""Canonical-only frontier extension; explicit terminal checks and spend caps."""
import argparse, contextlib, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
from providers import DirectProvider, MODELS
from run_guard import acquire_model_lock
ROOT=Path(__file__).resolve().parents[1];BENCH=ROOT/'runtime'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
with contextlib.redirect_stdout(sys.stderr):
 r=load('frontier_forked',BENCH/'scripts/experiment/forked_feedback_runner.py')
 side=load('frontier_sidecars',BENCH/'scripts/experiment/transfer_sidecar_runner.py')
p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--scope',choices=['smoke','pilot','canonical132'],default='pilot');p.add_argument('--cap-usd',type=float,required=True)
cli=p.parse_args();key=cli.model;output=ROOT/'private'/key;output.mkdir(parents=True,exist_ok=True);os.chmod(ROOT/'private',0o700)
try:run_lock=acquire_model_lock(output,key)
except RuntimeError as e:
 print(str(e),file=sys.stderr);sys.exit(3)
spec=r.ModelSpec(key=key,model_id=MODELS[key][0],temperature=.2 if key=='robotics' else None,reasoning_effort='medium',label=MODELS[key][0]);r.MODEL_SPECS[key]=spec
argv=sys.argv;sys.argv=['frontier','--provider','mock','--model',key,'--seed','2026','--conditions','full','neither','trace_status','--fixed-budget','8','--max-tokens','8192','--retain-truth-traces','--output-root',str(output/'episodes')]
args=r.parse_args();sys.argv=argv
assets=r.load_or_build_asset_index(r.load_json(args.manifest)['goals'],asset_index_path=args.asset_index,asset_roots=(BENCH,),rebuild=False)
goals=r.load_balanced_goals(args.manifest,asset_index=assets,budget_mode='fixed',fixed_budget=8)
canonical=[g for g in goals if g['source']=='canonical_world_gcond'];assert len(canonical)==132
pilot=[canonical[i]for i in [0,33,66,99]]
selected=pilot if cli.scope!='canonical132' else pilot+[g for g in canonical if g not in pilot]
bank=r.base.GoalBank(BENCH/'task_configs',args.goal_builder)
provider=DirectProvider(key,r.build_system_prompt('full'),output,cli.cap_usd)
manifest={'version':'frontier-canonical132-v1','model':spec.__dict__,'provider':provider.provider,'scope':cli.scope,'seed':2026,
 'selected_goals':[g['balanced_goal_id']for g in selected],'canonical_limit':132,'conditions':args.conditions,'attempts':8,'frames':32,'trace_states':32,
 'max_output_tokens':8192,'trace_output_tokens':16384,'effort':'medium','spend_cap_usd':cli.cap_usd,
 'legacy_first_success_absorption_corrected_for_hidden_status':True,'runner_sha256':hashlib.sha256((BENCH/'scripts/experiment/forked_feedback_runner.py').read_bytes()).hexdigest(),
 'provider_sha256':hashlib.sha256((ROOT/'scripts/providers.py').read_bytes()).hexdigest()}
manifest_path=output/f'{cli.scope}-manifest.json'
if manifest_path.exists():
 previous=manifest_path.read_bytes()
 if json.loads(previous)!=manifest:
  archive=output/'manifest-history';archive.mkdir(exist_ok=True)
  (archive/f'{cli.scope}-{hashlib.sha256(previous).hexdigest()}.json').write_bytes(previous)
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
if cli.scope=='smoke':
 g=pilot[0];targets=r.prediction_targets_for_goal(g,bank);scene_rules,dynamics=r.scene_and_tool_rules(g,bank)
 prompt=r.build_intro_prompt(g['goal_text'],8,targets,scene_rules=scene_rules,tool_dynamics=dynamics)
 text=provider.generate(history=[],prompt=prompt,image_paths=[Path(g['initial_screenshot'])])
 payload=r.parse_action_payload(text);coords=r.parse_action(text);valid=coords is not None
 if valid:valid=side.validate_predictions({**payload,'placement':list(coords)},targets=targets,required_placement=coords)['schema_valid']
 (output/'smoke-response.json').write_text(json.dumps({'response':text,'action_parse_valid':valid,'provider':provider.last_call_metadata},indent=2))
 print(json.dumps({'model':key,'api_call':'complete','action_parse_valid':valid,'estimated_spend_usd':provider.cumulative_cost_usd}),flush=True)
 sys.exit(0 if valid else 2)
status={'scope':cli.scope,'selected':len(selected),'main_complete':0,'sidecars_complete':0,'state':'running','partial_units':[]}
def save_status():
 status['estimated_spend_usd']=provider.cumulative_cost_usd;status['updated_unix']=time.time()
 temporary=output/'status.json.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(output/'status.json')
save_status()
for g in selected:
 try:
  provider.current_unit=g['balanced_goal_id'];provider.current_phase='main'
  print(json.dumps({'model':key,'goal':g['balanced_goal_id'],'phase':'main'}),flush=True)
  summary=r.run_unit(args=args,provider=provider,model_spec=spec,goal=g,goal_bank=bank,output_root=args.output_root)
  unit=r.goal_unit_dir(args.output_root,key,2026,g)
  # Completed files are insufficient if a format/geometry stop left a branch short.
  for condition in args.conditions:
   st=r.load_json(unit/'branches'/condition/'state.json');attempts=st.get('attempts',[])
   terminal=(len(attempts)==8) if condition=='neither' else (len(attempts)==8 or any(a.get('goal_succeeded')for a in attempts))
   if not st.get('completed') or not terminal:raise RuntimeError('Nonterminal branch checkpoint; investigate before extending.')
  status['main_complete']+=1;save_status()
  print(json.dumps({'model':key,'goal':g['balanced_goal_id'],'phase':'heldout_predictions'}),flush=True)
  provider.current_phase='heldout_predictions'
  result=side.run_unit_sidecars(args=args,provider=provider,goal=g,goal_bank=bank,unit_dir=unit,probe_cache=output/'probe-cache')
  if not side.sidecar_summary_complete(result,expected_conditions=args.conditions):raise RuntimeError('Held-out predictions are not terminal.')
  status['sidecars_complete']+=1;save_status()
 except Exception as e:
  status['state']='needs_attention';status['partial_units'].append({'goal':g['balanced_goal_id'],'reason':str(e)});save_status()
  print(json.dumps({'model':key,'state':'needs_attention','reason':str(e)}),flush=True);sys.exit(2)
status['state']='complete';save_status();print(json.dumps(status),flush=True)
