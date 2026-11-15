#!/usr/bin/env python3
"""Check every prospective initial image against a native simulator rendering.

This compares all RGB pixels, so it checks compound blue bodies as well as
red/green centers. Historical experiment images deliberately remain unchanged.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulator/runtime'))
import numpy as np
import pygame
from PIL import Image
from pyGameWorld import loadFromDict
from pyGameWorld.viewer import drawWorld


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,default=ROOT/'task_configs/asset_index_132_scene_corrected_2026-11-14.json')
    p.add_argument('--output',type=Path)
    a=p.parse_args();index=a.index.resolve();d=json.loads(index.read_text());rows=[]
    assert len(d['puzzles'])==132
    for key,e in sorted(d['puzzles'].items()):
        if not key.endswith('_downward'):continue
        layout=key.rsplit('_',1)[0];pair=d['puzzles'][layout+'_upward']
        assert e['world_sha256']==pair['world_sha256'] and e['screenshot_sha256']==pair['screenshot_sha256']
        world=(index.parent/e['world_path']).resolve();screenshot=(index.parent/e['screenshot_path']).resolve()
        assert hashlib.sha256(world.read_bytes()).hexdigest()==e['world_sha256']
        assert hashlib.sha256(screenshot.read_bytes()).hexdigest()==e['screenshot_sha256']
        w=loadFromDict(json.loads(world.read_text())['world'])
        expected=pygame.surfarray.array3d(drawWorld(w)).swapaxes(0,1)
        actual=np.array(Image.open(screenshot).convert('RGB'))
        assert actual.shape==expected.shape
        changed=int(np.any(expected!=actual,axis=2).sum())
        rows.append(dict(layout=layout,identical_rgb=changed==0,differing_pixels=changed,
            screenshot_sha256=e['screenshot_sha256'],world_sha256=e['world_sha256'],
            native_rgb_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
    report=dict(layouts=len(rows),direction_cells=2*len(rows),matching_layouts=sum(x['identical_rgb']for x in rows),
                index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),rows=rows)
    if a.output:
        if a.output.exists():raise ValueError('Output exists; use a new path')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    bad=[x['layout']for x in rows if not x['identical_rgb']]
    if bad:raise SystemExit('Native-render differences: '+', '.join(bad))
    print('PASS: all 66 layouts / 132 direction cells match native simulator rendering pixel-for-pixel.')


if __name__=='__main__':main()
