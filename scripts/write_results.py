#!/usr/bin/env python3
"""Write tables and a single concise results reference from computed estimates."""
import argparse,csv,json,shutil
from pathlib import Path
NAMES={'gpt5':'GPT-5','gemini31pro':'Gemini 3.1 Pro','qwen36plus':'Qwen 3.6 Plus','robotics_er2':'Gemini Robotics-ER 2','gpt':'GPT-5','gemini':'Gemini 3.1 Pro','qwen':'Qwen 3.6 Plus'}
POPULATIONS={'main':'Primary','broad':'Broad','canonical':'Canonical','transfer':'Novel probes','shared_first_failure_main':'First failure','shared_first_failure_transfer':'First failure/probes'}
MEASURES={'full_solve_rate':'Visual-feedback success','neither_solve_rate':'Retry success','trace_status_solve_rate':'JSON-feedback success','initial_success':'First-action success','chosen_pq_initial':'Chosen PQ, initial','chosen_pq_final':'Chosen PQ, final','chosen_pq_gain':'Chosen PQ gain','subsequent_solve':'Later success difference','next_chosen_pq':'Next chosen-PQ difference','early_novel_pq':'One-rollout novel-PQ difference','novel_pre':'Novel PQ, before','novel_post':'Novel PQ, after','novel_gain':'Novel PQ gain','novel_gain_beyond_retry':'Novel gain beyond retry','figure4_A':'Success / chosen PQ','figure4_B':'Chosen / novel PQ, basic','figure4_B_richer':'Chosen / novel PQ, richer','figure4_C':'Success / early novel PQ','full_minus_neither_solve':'Visual minus retry','trace_status_minus_neither_solve':'JSON minus retry','trace_status_minus_full_solve':'JSON minus visual','status_only_solve_rate':'Status-only success','frames_only_solve_rate':'Frames-only success','full_minus_status_only':'Visual minus status only','status_only_minus_neither':'Status only minus retry','frames_only_minus_neither':'Frames only minus retry','full_minus_frames_only':'Visual minus frames only'}
def ci(s,pct=False):
 f=100 if pct else 1;digits=2 if pct else 3
 return f"{s['estimate']*f:.{digits}f} [{s['ci95'][0]*f:.{digits}f}, {s['ci95'][1]*f:.{digits}f}]"
def generate(out):
 S=json.loads((out/'summary.json').read_text());P=json.loads((out/'populations.json').read_text());rows=list(csv.DictReader((out/'estimates.csv').open()));tables=out/'tables';tables.mkdir(exist_ok=True)
 groups={'main':[r for r in rows if r['population']=='main'and r['metric']in ['full_solve_rate','initial_success','chosen_pq_initial','chosen_pq_final','chosen_pq_gain']], 'feedback':[r for r in rows if r['population'].startswith('shared_first_failure')], 'feedback_decomposition':[r for r in rows if r['population']=='canonical'],'json_controls':[r for r in rows if r['population']in ['main','broad']and ('trace_status'in r['metric']or r['metric']in ['full_solve_rate','neither_solve_rate','full_minus_neither_solve'])],'correspondence':[r for r in rows if r['metric'].startswith('figure4_')],'transfer':[r for r in rows if r['population']=='transfer'and not r['metric'].startswith('figure4_')]}
 for name,rr in groups.items():
  with (tables/(name+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
  tex=['\\begin{tabular}{lllrr}','\\toprule','Model & Population & Measure & Tasks & Estimate [95\\% CI] \\\\','\\midrule']
  for r in rr:
   measure=MEASURES.get(r['metric'],r['metric'].replace('_',' '));population=POPULATIONS[r['population']];tex.append(f"{NAMES[r['model']]} & {population} & {measure} & {r['n_tasks']} & {float(r['estimate']):.3f} [{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}] \\\\")
  tex+=['\\bottomrule','\\end{tabular}'];(tables/(name+'.tex')).write_text('\n'.join(tex)+'\n')
 sensitivity=list(csv.DictReader((out/'pq_scale_sensitivity.csv').open()));shutil.copyfile(out/'pq_scale_sensitivity.csv',tables/'pq_scale_sensitivity.csv')
 labels={'figure4A':'Success / chosen PQ','next_chosen_after_failure':'Chosen-PQ feedback benefit','early_novel_after_failure':'Early novel-PQ feedback benefit','terminal_novel_gain_beyond_retry':'Novel gain beyond retry'}
 tex=['\\begin{tabular}{lrlrr}','\\toprule','Model & $\\sigma$ (px) & Measure & Tasks & Estimate [95\\% CI] \\\\','\\midrule']
 for r in sensitivity:tex.append(f"{NAMES[r['model']]} & {r['sigma_pixels']} & {labels[r['metric']]} & {r['tasks']} & {float(r['estimate']):.3f} [{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}] \\\\")
 tex+=['\\bottomrule','\\end{tabular}'];(tables/'pq_scale_sensitivity.tex').write_text('\n'.join(tex)+'\n')
 text=['# Finalized model results','',f"The benchmark contains 1,692 goals on 66 layouts with two tool directions. The reported common populations contain **{len(P['main'])} main tasks**, **{len(P['broad'])} broad-control tasks**, and **{len(P['matched_transfer'])} matched transfer tasks**. Exact task IDs are in `populations.json`. Robotics-ER 2 is a separate canonical extension. All intervals below are 95% layout-cluster bootstrap intervals. They describe variation across layouts for the recorded runs, not variation across repeated model seeds.",'','## Interaction','','| Model | First action success (%) | Final episode success (%) | Chosen PQ, first | Chosen PQ, final | Novel PQ, pre | Novel PQ, post |','|---|---:|---:|---:|---:|---:|---:|']
 for m in ['gpt5','gemini31pro','qwen36plus','robotics_er2']:
  r=S[m];i=r['interaction'];text.append('| '+NAMES[m]+' | '+' | '.join([ci(i['initial_success'],True),ci(r['solve_rates']['main']['full'],True),ci(i['chosen_pq_initial']),ci(i['chosen_pq_final']),ci(i['novel_pre']),ci(i['novel_post'])])+' |')
 if P['removed_from_historical_transfer']:
  text+=['','Matched transfer eligibility also requires that frozen probes remain untried and that all required queries are available. The following tasks are omitted from the common transfer population only:','']
  for goal,reasons in sorted(P['removed_from_historical_transfer'].items()):text.append('- `'+goal+'`: '+'; '.join(NAMES[m]+': '+', '.join(rr)for m,rr in reasons.items()if rr)+'.')
 text+=['','Initial/final slopes are descriptive: actions can change, retries occur, and visible-success branches stop. Novel pre/post predictions concern the same fixed intervention.','', '## Feedback beyond retry','','Only episodes sharing the same failed first action enter these comparisons. Early novel probes are compared after exactly one rollout.','', '| Model | Later success difference | Next chosen-PQ difference | Early novel-PQ difference |','|---|---:|---:|---:|']
 for m in ['gpt5','gemini31pro','qwen36plus','robotics_er2']:
  r=S[m]['feedback'];text.append('| '+NAMES[m]+' | '+' | '.join(ci(r[k])+f"; n={r[k]['n']}"for k in ['subsequent_solve','next_chosen_pq','early_novel_pq'])+' |')
 text+=['','## Correspondence','','| Model | Action success / chosen PQ (adjusted difference) | Chosen / novel PQ (richer partial Spearman) | First-action success / early novel PQ (adjusted difference) |','|---|---:|---:|---:|']
 for m in ['gpt5','gemini31pro','qwen36plus']:
  f=S[m]['figure4'];text.append('| '+NAMES[m]+' | '+' | '.join(ci(f[k])for k in ['A','B_richer','C'])+' |')
 text+=['',f"Robotics-ER 2: {S['robotics_er2']['solve_rates']['main']['full']['numerator']}/{len(P['robotics_main'])} solved; Figure 4A estimator {ci(S['robotics_er2']['figure4']['A'])}. Its own valid novel-probe population contains {len(P['robotics_transfer'])} tasks.",'','A controls exact goal and attempt with equal episode weights. B controls layout, direction, initial chosen prediction, baseline novel prediction and feedback exposure. C controls baseline probe quality, layout and direction at the common one-rollout checkpoint. These are conditional associations, not causal effects of success.','', '## Corrected-scene VQA','','| Model | Correct / eligible | Accuracy | 95% layout-bootstrap CI |','|---|---:|---:|---:|']
 for m in ['gpt','gemini','qwen']:
  s=S['vqa'][m];text.append(f"| {NAMES[m]} | {s['correct']}/{s['eligible']} | {100*s['accuracy']:.2f}% | [{100*s['ci95_low']:.2f}, {100*s['ci95_high']:.2f}]% |")
 text+=['','The same frozen compatible-format control is used: JSON-object output, low reasoning effort, 16,384-token cap, seed 2026, and temperature 0.2 where supported. Nine calls replace the three corrected scenes; 189 previously returned calls are preserved. Missing/invalid answers receive no credit; the original 60-pixel divider rule controls quadrant eligibility.','', '## Tables and figures','','Vector Figures 3 and 4 are in `figures/`. Every plotted estimate and interval is recorded in `figures/plotted_values.json`. Machine-readable appendix tables and standalone LaTeX tabular fragments are in `tables/`; they require the `booktabs` package. Structured-state JSON controls supply observable object positions/states rather than rollout images and have an independent initial action.','', '## Reporting statement','','“We corrected inconsistent initial visual states in three layouts and reran their affected model episodes and fixed prediction probes. Reported model results combine these corrected observations with unchanged records for all other tasks.”','','Human participant-level data remain private. Human estimates and hypothetical human curves are not included in this model-only release. Historical controls and figures in the archive retain their original populations and must not be substituted for these finalized estimates.']
 (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
 captions=f'''# Figure captions

**Figure 3. Performance during interaction and the benefit of feedback.** A: Initial and final performance under full visual feedback, on {len(P['main'])} common tasks for action and chosen-action prediction and {len(P['matched_transfer'])} matched tasks for novel prediction. Novel pre/post predictions use the same untried placement. B: Full-feedback minus retry-only performance after the same failed first action; novel predictions are compared after exactly one rollout. Points are means; intervals are 95% layout-bootstrap confidence intervals. Initial-to-final changes alone do not identify a causal feedback benefit.

**Figure 4. Correspondence depends on the ability pair and observation checkpoint.** Top: Observed prediction quality by action outcome (A, C), and terminal chosen/novel prediction pairs with descriptive linear fits (B). Bottom: A, chosen-PQ difference controlling exact goal and attempt, with equal episode weights; B, partial rank correlations controlling layout/direction (filled) and additionally baseline predictions and rollout count (open); C, baseline-adjusted novel-PQ difference after exactly one rollout, controlling layout/direction. A uses {len(P['main'])} tasks; B-C use {len(P['matched_transfer'])} matched tasks. Intervals are 95% layout-bootstrap confidence intervals. These are conditional associations, not causal effects of success.
'''
 (out/'figure_captions.md').write_text(captions)
 (tables/'README.md').write_text('''# Appendix and results tables

Each CSV contains exact estimates, interval bounds, task counts, cluster counts, and bootstrap settings. Matching LaTeX files are standalone `booktabs` tabular fragments, not manuscript edits.

- `main`: first/final success and chosen-action prediction quality.
- `feedback`: paired full-feedback minus retry contrasts after a shared failed first action; novel probes are compared after one rollout.
- `feedback_decomposition`: canonical visual, frames-only, status-only, retry, and JSON conditions, with paired task-level contrasts.
- `json_controls`: primary and broad populations are explicitly separated; JSON uses an independent initial action.
- `correspondence`: Figure 4A/C are adjusted PQ differences; B is a partial Spearman correlation, with basic and richer controls listed separately.
- `transfer`: fixed novel-placement pre/post quality, gain, and gain beyond retry.
- `pq_scale_sensitivity`: the same finalized populations under Gaussian scales 75, 100, and 150 pixels. This supplementary check preserves its historical bootstrap settings (10,000 draws for mean contrasts); primary Figure 3 uses 2,000 draws. Point estimates agree at scale 150, while interval endpoints can differ slightly from Monte Carlo sampling.

Success rates are fractions; success differences are probability differences, not relative percentages. PQ is graded spatial similarity, not percentage accuracy. All intervals are 95% whole-layout bootstrap intervals. Robotics-ER 2 uses its own canonical population; task counts distinguish it from the three main models. See `../populations.json` and `../method.json` for exact membership and estimators.
''')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();generate(a.results)
