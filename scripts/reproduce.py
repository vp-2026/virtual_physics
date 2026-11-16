#!/usr/bin/env python3
"""One offline command for integrity checks, tables, and vector Figures 3/4."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/evaluation'),str(ROOT/'scripts/release'),str(ROOT/'scripts')]
from verify_release import verify
from analyze_finalized import analyze
from reproduce_vqa import reproduce
from write_results import generate
from verify_results import verify as verify_numerics
from baselines import analyze_baselines
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/reproduction');a=p.parse_args();out=a.output.resolve()
validation=verify(geometry=True);analyze(ROOT,out);vqa=reproduce(out/'vqa')
summary=json.loads((out/'summary.json').read_text());summary['vqa']=vqa;(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
verify_numerics(ROOT,out)
analyze_baselines(ROOT,out,out/'baselines')
subprocess.run([sys.executable,str(ROOT/'scripts/render_figures.py'),'--results',str(out),'--output',str(out/'figures')],check=True);generate(out)
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print('Complete. Tables, figures, populations and results:',out)
