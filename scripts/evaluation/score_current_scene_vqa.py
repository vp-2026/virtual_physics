#!/usr/bin/env python3
"""Score every planned current-scene item; missing answers count as incorrect."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from vqa_io import write_json, write_jsonl, sha256_file
from score_static_vqa import read_jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument('--battery',type=Path,required=True)
    p.add_argument('--responses',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new/empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    for path,h in json.loads((a.battery/'freeze_sha256.json').read_text()).items():
        assert sha256_file(a.battery/path)==h,path
    history=read_jsonl(a.responses) if a.responses.exists() else []
    planned=read_jsonl(a.battery/'call_manifest.jsonl');selected=[];status=[]
    assert set(r['call_id'] for r in history)<=set(c['call_id'] for c in planned)
    for c in planned:
        rr=[r for r in history if r['call_id']==c['call_id']]
        accepted=[r for r in rr if 'error' not in r]
        # First returned answer, including a malformed answer, is the outcome.
        row=accepted[0] if accepted else {**c,'format_valid':False,'parsed_response':{},'error':'missing_or_transport_failure'}
        selected.append(row);status.append(dict(call_id=c['call_id'],model_key=c['model_key'],
            attempts=len(rr),missing=not accepted,format_valid=bool(row.get('format_valid'))))
    write_jsonl(out/'planned_responses.jsonl',selected)
    subprocess.run([sys.executable,str(Path(__file__).with_name('score_static_vqa.py')),
        str(out/'planned_responses.jsonl'),'--questions',str(a.battery/'questions.jsonl'),
        '--coordinate-ground-truth',str(a.battery/'ground_truth.jsonl'),'--output-dir',str(out)],check=True)
    write_json(out/'completion_report.json',dict(planned_calls=len(planned),raw_requests=len(history),
        missing_calls=sum(x['missing'] for x in status),invalid_calls=sum(not x['format_valid'] for x in status),calls=status))


if __name__=='__main__':main()
