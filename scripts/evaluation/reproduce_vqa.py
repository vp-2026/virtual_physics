#!/usr/bin/env python3
"""Score the finalized VQA responses with the frozen question/scoring rules."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT/'archive/pre_correction_2026-11-14'
sys.path.insert(0,str(LEGACY/'vqa/scripts'))
from summarize_current_scene_vqa import aggregate

def reproduce(out):
 data=ROOT/'data/vqa';subprocess.run([sys.executable,str(LEGACY/'vqa/scripts/score_current_scene_vqa.py'),'--battery',str(data/'battery'),'--responses',str(data/'responses.jsonl'),'--output',str(out)],check=True)
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
