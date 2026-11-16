#!/usr/bin/env python3
"""Score the finalized VQA responses with the frozen question/scoring rules."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import numpy as np

def aggregate(rows):
    labs=sorted({r['layout_id'] for r in rows})
    num=np.array([sum(int(r['correct']) for r in rows if r['layout_id']==l) for l in labs])
    den=np.array([sum(r['layout_id']==l for r in rows) for l in labs])
    ix=np.random.default_rng(20261114).integers(len(labs),size=(10000,len(labs)))
    ci=np.quantile(num[ix].sum(1)/den[ix].sum(1),[.025,.975])
    return dict(correct=int(num.sum()),eligible=int(den.sum()),accuracy=float(num.sum()/den.sum()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),layouts=len(labs))

def reproduce(out):
 data=ROOT/'data/vqa';subprocess.run([sys.executable,str(ROOT/'scripts/evaluation/score_current_scene_vqa.py'),'--battery',str(data/'battery'),'--responses',str(data/'responses.jsonl'),'--output',str(out)],check=True)
 report_path=out/'score_report.json';report=json.loads(report_path.read_text())
 for key in ['accuracy_summary','question_scores']:report[key]=Path(report[key]).name
 report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 for path in out.glob('*.csv'):path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
 rows=list(csv.DictReader((out/'question_scores.csv').open()))
 for r in rows:assert int(r['correct'])==int(bool(r['observed'])and r['observed']==r['expected'])
 result={}
 for model in sorted({r['model_key']for r in rows}):
  rr=[r for r in rows if r['model_key']==model and r['primary_scored']=='1'];result[model]=aggregate(rr)
  result[model].update(bootstrap_draws=10000,bootstrap_seed=20261114)
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(reproduce(a.output),indent=2))
