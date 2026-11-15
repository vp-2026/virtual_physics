#!/usr/bin/env python3
"""Summarize frozen VQA scores with missing responses retained in denominators.

Reports primary strict-schema and separate Qwen interface diagnostic without
substituting answers between conditions. No inference or score mutation.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def read(path):
    return list(csv.DictReader(path.open()))

def write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def aggregate(rows):
    labs=sorted({r['layout_id'] for r in rows})
    num=np.array([sum(int(r['correct']) for r in rows if r['layout_id']==l) for l in labs])
    den=np.array([sum(r['layout_id']==l for r in rows) for l in labs])
    ix=np.random.default_rng(20261114).integers(len(labs),size=(10000,len(labs)))
    ci=np.quantile(num[ix].sum(1)/den[ix].sum(1),[.025,.975])
    return dict(correct=int(num.sum()),eligible=int(den.sum()),accuracy=float(num.sum()/den.sum()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),layouts=len(labs))

def main():
    p=argparse.ArgumentParser();p.add_argument('--battery',type=Path,required=True)
    p.add_argument('--primary-scores',type=Path,required=True)
    p.add_argument('--interface-scores',type=Path,required=True)
    p.add_argument('--compatible-scores',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a new/empty output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    sources={'primary_strict_schema':read(a.primary_scores/'question_scores.csv'),
             'qwen_json_object_diagnostic':read(a.interface_scores/'question_scores.csv'),
             'all_model_compatible_condition':read(a.compatible_scores/'question_scores.csv'),
             'historical_image':read(ROOT/'vqa/static_scene/results/scored/question_scores.csv')}
    sources={k:[r for r in v if r['input_condition']=='image_only'] for k,v in sources.items()}
    # Independently recompute every scored item from the saved letters/keys.
    for rows in sources.values():
        for r in rows:assert int(r['correct'])==int(bool(r['observed']) and r['observed']==r['expected'])
    changed={q['question_id'] for q in json.loads((a.battery/'question_changes.json').read_text())}
    eligible={k:[r for r in v if r['primary_scored']=='1'] for k,v in sources.items()}
    summaries=[];categories=[];sens=[];comparison=[];unchanged=[]
    for phase,rows in eligible.items():
        for model in sorted({r['model_key'] for r in rows}):
            rr=[r for r in rows if r['model_key']==model]
            summaries.append(dict(phase=phase,model=model,**aggregate(rr)))
            for category in sorted({r['category'] for r in rr}):
                categories.append(dict(phase=phase,model=model,category=category,**aggregate([r for r in rr if r['category']==category])))
            if phase!='historical_image':
                ss=[r for r in rr if r['layout_id'] not in ('BackUp_8','Balance_8')]
                sens.append(dict(phase=phase,model=model,restriction='exclude_two_image_world_mismatches',**aggregate(ss)))
                # Descriptive available-answer score, never a full-panel estimate.
                ss=[r for r in rr if r['format_valid']=='1']
                if ss:sens.append(dict(phase=phase,model=model,restriction='format_valid_scenes_only_selection_biased',**aggregate(ss)))
                hist=[r for r in eligible['historical_image'] if r['model_key']==model]
                h=aggregate(hist);n=aggregate(rr)
                comparison.append(dict(phase=phase,model=model,historical_correct=h['correct'],historical_eligible=h['eligible'],historical_accuracy=h['accuracy'],current_correct=n['correct'],current_eligible=n['eligible'],current_accuracy=n['accuracy'],difference=n['accuracy']-h['accuracy']))
                hh={r['question_id']:r for r in hist};nn={r['question_id']:r for r in rr}
                ids=sorted((hh.keys()&nn.keys())-changed)
                x,y=aggregate([hh[i] for i in ids]),aggregate([nn[i] for i in ids])
                unchanged.append(dict(phase=phase,model=model,eligible=len(ids),historical_correct=x['correct'],current_correct=y['correct'],historical_accuracy=x['accuracy'],current_accuracy=y['accuracy'],difference=y['accuracy']-x['accuracy']))
    for name,rows in [('accuracy_summary',summaries),('category_accuracy',categories),('sensitivity',sens),('historical_comparison',comparison),('unchanged_question_comparison',unchanged)]:write(a.output/(name+'.csv'),rows)
    completion={}
    for phase,path in [('primary_strict_schema',a.primary_scores),('qwen_json_object_diagnostic',a.interface_scores),('all_model_compatible_condition',a.compatible_scores)]:
        c=json.loads((path/'completion_report.json').read_text());counts={}
        for m in sorted({r['model_key'] for r in c['calls']}):
            rr=[r for r in c['calls'] if r['model_key']==m]
            counts[m]=dict(planned=len(rr),transport_missing=sum(r['missing'] for r in rr),format_valid=sum(r['format_valid'] for r in rr),format_invalid=sum(not r['format_valid'] for r in rr))
            items=[r for r in sources[phase] if r['model_key']==m]
            counts[m]['eligible_items_without_answer']=sum(r['primary_scored']=='1' and not r['observed'] for r in items)
            layouts={r['layout_id'] for r in items}
            counts[m]['calls_without_any_scored_answer']=sum(not any(r['observed'] for r in items if r['layout_id']==l and r['primary_scored']=='1') for l in layouts)
        completion[phase]=counts
    (a.output/'completion_summary.json').write_text(json.dumps(completion,indent=2)+'\n')
    (a.output/'methods.json').write_text(json.dumps(dict(bootstrap_draws=10000,bootstrap_seed=20261114,cluster='visual layout',interval='percentile',missing='incorrect in fixed eligible denominator',diagnostic='Separate post-primary condition; no substitution',caveat='Format-valid-only accuracy is selected and cannot estimate all-scene perception. Historical/current keys and provider versions differ; not a controlled asset effect.'),indent=2)+'\n')
    print(json.dumps(completion,indent=2))

if __name__=='__main__':main()
