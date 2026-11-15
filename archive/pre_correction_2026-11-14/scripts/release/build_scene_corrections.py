#!/usr/bin/env python3
"""Versioned scene corrections; never replaces archived experiment inputs.

Render directly from the simulator worlds that were actually used for scoring.
The corrected index is for future experiments, not a description of old runs.
"""
import argparse
import copy
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
import pymunk
from pyGameWorld import loadFromDict
from pyGameWorld.viewer import drawWorld
from PIL import Image

LAYOUTS={'BackUp_8':'013','Balance_8':'031','Falling_1':'063'}
INDEX='task_configs/asset_index_132_scene_corrected_2026-11-14.json'
MANIFEST='analysis/scene_mismatch_2026-11-14/corrected_asset_manifest.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def color_center(pixels,color):
    x=pixels.astype(float)
    if color=='green':mask=(x[:,:,1]>1.1*x[:,:,0])&(x[:,:,1]>1.1*x[:,:,2])&(x[:,:,1]>80)
    else:mask=(x[:,:,0]>1.3*x[:,:,1])&(x[:,:,0]>1.3*x[:,:,2])&(x[:,:,0]>120)
    yy,xx=np.where(mask)
    if not len(xx):raise ValueError('No '+color+' pixels')
    return np.array([(xx.min()+xx.max())/2,len(x)-(yy.min()+yy.max())/2])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    a=parser.parse_args(); old=ROOT/'task_configs/asset_index_132.json'
    index=json.loads(old.read_text());corrected=copy.deepcopy(index);records=[]
    if not a.check and ((ROOT/INDEX).exists() or (ROOT/MANIFEST).exists()):
        raise ValueError('Correction already exists; use --check')
    for layout,cell in LAYOUTS.items():
        entry=index['puzzles'][layout+'_downward']
        world=(old.parent/entry['world_path']).resolve()
        historic=(old.parent/entry['screenshot_path']).resolve()
        assert sha(world)==entry['world_sha256'] and sha(historic)==entry['screenshot_sha256']
        d=json.loads(world.read_text());w=loadFromDict(d['world']);surface=drawWorld(w)
        pixels=pygame.surfarray.array3d(surface).swapaxes(0,1)
        target=ROOT/f'132_base_environments/corrections/2026-11-14/{layout}/initial_scene.png'
        if a.check:
            actual=np.array(Image.open(target).convert('RGB'))
            assert np.array_equal(pixels,actual),(layout,'rendered pixels differ')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                assert np.array_equal(np.array(Image.open(target).convert('RGB')),pixels), 'Refusing to overwrite different pixels'
            else:pygame.image.save(surface,str(target))
        goal_name=next(k for k,v in d['world']['objects'].items() if v.get('innerColor')=='green')
        expected_green=np.mean(np.array(d['world']['objects'][goal_name]['points']),axis=0)
        expected_red=np.array(d['world']['objects']['Ball']['position'])
        for color,expected in [('green',expected_green),('red',expected_red)]:
            assert np.max(np.abs(color_center(pixels,color)-expected))<=3,(layout,color)
        for direction in ['downward','upward']:
            key=layout+'_'+direction;raw=index['puzzles'][key]
            assert raw['world_sha256']==entry['world_sha256']
            assert raw['screenshot_sha256']==entry['screenshot_sha256']
            corrected['puzzles'][key]['screenshot_path']='../'+str(target.relative_to(ROOT))
            corrected['puzzles'][key]['screenshot_sha256']=sha(target)
        records.append(dict(layout=layout,directions=['downward','upward'],
            historical_screenshot=str(historic.relative_to(ROOT)),historical_screenshot_sha256=sha(historic),
            authoritative_simulation_world=str(world.relative_to(ROOT)),world_sha256=sha(world),
            corrected_screenshot=str(target.relative_to(ROOT)),corrected_screenshot_sha256=sha(target),
            corrected_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
            simulator_green_center=expected_green.tolist(),rendered_green_center=color_center(pixels,'green').tolist(),
            simulator_red_center=expected_red.tolist(),rendered_red_center=color_center(pixels,'red').tolist(),
            native_vertices_match_world=bool(np.allclose(np.asarray(w.objects[goal_name].vertices),d['world']['objects'][goal_name]['points']))))
    corrected['provenance']='Prospective scene-corrected index, 2026-11-14. Three images (BackUp_8, Balance_8, Falling_1) re-rendered from the executed simulator worlds. NOT the inputs used in historical experiments. Original index and records remain frozen.'
    corrected['historical_index_sha256']=sha(old)
    corrected['correction_manifest']='../'+MANIFEST
    manifest=dict(schema_version=1,date='2026-11-14',historical_index=str(old.relative_to(ROOT)),
        historical_index_sha256=sha(old),corrected_index=INDEX,corrected_images=records,
        renderer='pyGameWorld.loadFromDict + pyGameWorld.viewer.drawWorld, no simulation steps',
        renderer_sources={p:sha(ROOT/p)for p in ['scripts/release/build_scene_corrections.py','simulator/runtime/pyGameWorld/world.py','simulator/runtime/pyGameWorld/object.py','simulator/runtime/pyGameWorld/viewer/__init__.py']},
        pygame_version=pygame.version.ver,pymunk_version=pymunk.version,
        frozen_worlds_changed=False,frozen_screenshots_changed=False,frozen_results_changed=False)
    if a.check:
        assert json.loads((ROOT/INDEX).read_text())==corrected,'Corrected index differs'
        expected=json.loads((ROOT/MANIFEST).read_text())
        assert expected['corrected_images']==records,'Correction manifest differs'
        for path,h in expected['renderer_sources'].items():assert sha(ROOT/path)==h,(path,'renderer changed')
    else:
        (ROOT/INDEX).write_text(json.dumps(corrected,indent=2,sort_keys=True)+'\n')
        (ROOT/MANIFEST).parent.mkdir(parents=True,exist_ok=True)
        (ROOT/MANIFEST).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    for key,entry in corrected['puzzles'].items():
        for field in ['world','screenshot']:
            assert sha((old.parent/entry[field+'_path']).resolve())==entry[field+'_sha256'],(key,field)
    print('PASS: 132 index entries SHA-verified; 3 corrected images / 6 direction cells; native-render pixels and red/green geometry agree; historical assets preserved.')


if __name__=='__main__':main()
