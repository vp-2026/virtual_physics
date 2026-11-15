#!/usr/bin/env python3
"""Freeze a new image-only VQA battery on the exact released current scenes.

No inference. Original question templates, option balancing, and divider
exclusion rule are preserved. Scene-dependent answers/options are regenerated
before inference; every changed item is recorded. Historical files are read-only.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
from generate_vqa import (read_json, write_json, write_jsonl, sha256_file,
    questions_for_scene, visible_objects, role_objects, numeric,
    build_user_prompt, system_prompt)
from generate_coordinate_vqa import center_of_geometry, image_color_center

ROOT = Path(__file__).resolve().parents[2]


def generate(repo, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be new/empty; never overwrite a frozen test set")
    index = read_json(repo/'task_configs/asset_index_132.json')['puzzles']
    old = [json.loads(x) for x in (repo/'vqa/static_scene/questions.jsonl').read_text().splitlines()]
    coord = [json.loads(x) for x in (repo/'vqa/coordinate_localization/ground_truth.jsonl').read_text().splitlines()]
    old = {r['layout_id']: r['questions'] for r in old}
    qs, assets, truth, calls, changes, annotations = [], [], [], [], [], []
    note = ('The attached clean 600-by-600 image contains the initial scene. '
            'Image coordinates use (0,0) at bottom-left; x increases to the '
            'right and y increases upward.')
    for i, prior in enumerate(coord):
        layout = prior['layout_id']
        a, b = [index[layout+'_'+d] for d in ('downward', 'upward')]
        assert a['screenshot_sha256'] == b['screenshot_sha256']
        assert a['world_sha256'] == b['world_sha256']
        image = (repo/'task_configs'/a['screenshot_path']).resolve()
        worldpath = (repo/'task_configs'/a['world_path']).resolve()
        assert sha256_file(image) == a['screenshot_sha256']
        assert sha256_file(worldpath) == a['world_sha256']
        world = read_json(worldpath)
        red, green, _ = role_objects(visible_objects(world))
        # Pre-inference visual audit found two mismatched goal locations.
        # For these screenshots alone, annotate the visible green interior's
        # bounding rectangle; never alter the released simulator world. These
        # items ask only coarse relations/containment, far from its boundary.
        if layout in ('BackUp_8','Balance_8'):
            from PIL import Image
            im=Image.open(image).convert('RGB');pixels=[]
            for y in range(im.height):
                for x in range(im.width):
                    r,g,bl=im.getpixel((x,y))
                    if g>r+20 and g>bl+20 and g>90:pixels.append((x,599-y))
            xmin,xmax=min(x for x,y in pixels),max(x for x,y in pixels)
            ymin,ymax=min(y for x,y in pixels),max(y for x,y in pixels)
            original=copy.deepcopy(green)
            green['points']=[[xmin,ymax],[xmin,ymin],[xmax,ymin],[xmax,ymax]]
            annotations.append(dict(layout_id=layout,object='green goal',original_world_geometry=original,
                vqa_only_geometry=copy.deepcopy(green),method='Visible green interior bounding rectangle; independently visually inspected before inference',
                scope='Question-key geometry only. Original screenshot/world/simulation unchanged.'))
        questions = questions_for_scene(layout, i, world)
        assert len(questions) == 11
        qs.append(dict(layout_id=layout, questions=questions))
        for before, after in zip(old[layout], questions):
            assert before['question_id'] == after['question_id']
            if before != after:
                changes.append(dict(layout_id=layout, question_id=after['question_id'],
                    changed_fields=[k for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)],
                    historical=before, current=after))
        targets = {}
        for key, obj, color in [('red_ball_center',red,'red'), ('green_container_center',green,'green')]:
            center = center_of_geometry(obj)
            pixel = image_color_center(image, color)
            distance = sum((x-y)**2 for x,y in zip(center,pixel))**.5
            if distance > 3:
                raise ValueError(f'{layout}: world/image center mismatch {distance}')
            targets[key] = dict(center=center,pixel_mask_center=pixel,
                world_pixel_center_disagreement_px=distance,
                combined_halves_scored=abs(center[0]-300)>60 and abs(center[1]-300)>60)
        truth.append(dict(layout_id=layout,targets=targets))
        # Relative paths make the frozen battery relocatable. Runner resolves
        # them against the battery root, never against an archived home path.
        image_rel = f'images/{layout}.png'
        (out/'images').mkdir(parents=True,exist_ok=True)
        shutil.copyfile(image,out/image_rel)
        payload = dict(system=system_prompt(),user_text=build_user_prompt(questions,note,numeric(green.get('density'))>0),
            image_path=image_rel,scene_json=None,response_schema=dict(type='object',
            required_question_ids=[q['question_id'] for q in questions],values='one option letter'))
        payload_rel = f'payloads/{layout}.json'
        write_json(out/payload_rel,payload)
        assets.append(dict(layout_id=layout,image_path=image_rel,image_sha256=sha256_file(image),
            world_path=str(worldpath.relative_to(repo)),world_sha256=sha256_file(worldpath),
            historical_image_sha256=prior['source_image_sha256'],
            image_matches_historical=sha256_file(image)==prior['source_image_sha256']))
        for model in ('gpt','gemini','qwen'):
            calls.append(dict(call_id=f'current_scene_20261114:{layout}:{model}:image_only',
                layout_id=layout,model_key=model,input_condition='image_only',
                payload_path=payload_rel,benchmark_cell_aliases=[layout+'_downward',layout+'_upward']))
    assert len(assets)==66 and len(calls)==198
    for name,rows in [('questions',qs),('assets',assets),('ground_truth',truth),('call_manifest',calls)]:
        write_jsonl(out/(name+'.jsonl'),rows)
    write_json(out/'question_changes.json',changes)
    write_json(out/'image_world_annotation_exceptions.json',annotations)
    eligible = sum(9+sum(t['combined_halves_scored'] for t in r['targets'].values()) for r in truth)
    metadata = dict(layouts=66,questions=726,eligible_questions_per_model=eligible,
        excluded_quadrants_per_model=726-eligible,planned_calls=198,seed=2026,
        input_condition='image_only',changed_question_records=len(changes),
        image_world_mismatches=['BackUp_8','Balance_8'],
        annotation_policy='Two VQA-only green goal rectangle annotations from screenshot color masks, frozen before inference; report a sensitivity excluding both layouts.',
        historical_json_qwen='Unchanged: 565/677; this run has no JSON arm.',
        scorer='score_current_scene_vqa.py; fixed eligible denominator, missing answers incorrect',
        model_settings=dict(reasoning_effort='medium',max_tokens=8192,seed=2026,
            temperature='0.2 for Gemini/Qwen; omitted for GPT-5 (unsupported)'),
        comparison_caveat='Scene-dependent keys/options are regenerated. Medium reasoning and8192 tokens match actual archived historical response records, not CLI defaults. Provider versions may differ; not a controlled asset-only effect.',
        protocol_revision='v2: supersedes an incomplete low/2048 pilot after output-limit failures; no pilot answer is substituted into this full rerun.')
    write_json(out/'protocol.json',metadata)
    files={str(p.relative_to(out)):sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'freeze_sha256.json',files)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(generate(a.repo_root.resolve(),a.output.resolve()),indent=2))
