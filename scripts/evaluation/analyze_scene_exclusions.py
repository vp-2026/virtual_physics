#!/usr/bin/env python3
"""Sensitivity to the two documented initial-image/world mismatches.

Reads frozen model archives only; never calls a provider or reads human data.
Original task memberships and estimator functions are retained, then filtered
by layout (both directions). Outputs are new and must not already exist.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import collections
import hashlib
import json
import csv
from pathlib import Path
import numpy as np
from analyze_figure4_correspondence import reconstruct, observations, conditional_mean, MODELS
import figure4_estimators as est

EXCLUDE = {'BackUp_8', 'Balance_8'}


def bootstrap(values, layouts, seed=20261114, draws=100000):
    group = collections.defaultdict(list)
    for value, layout in zip(values, layouts):
        group[layout].append(float(value))
    labels = sorted(group)
    sums = np.array([sum(group[k]) for k in labels])
    counts = np.array([len(group[k]) for k in labels])
    ix = np.random.default_rng(seed).integers(len(labels), size=(draws, len(labels)))
    bs = sums[ix].sum(axis=1) / counts[ix].sum(axis=1)
    return dict(estimate=float(np.mean(values)), ci95=np.quantile(bs, [.025, .975]).tolist(),
                n=len(values), layouts=len(labels), bootstrap_draws=draws, bootstrap_seed=seed)


def label_seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], 'big')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--exclude-layouts', nargs='+', default=sorted(EXCLUDE),
                   help='Default: the two requested layouts; add Falling_1 for the expanded integrity sensitivity')
    p.add_argument('--verified-reconstruction', type=Path,
                   help='Optional cache produced by analyze_figure4_correspondence.py')
    a = p.parse_args(); repo = a.repo_root.resolve(); out = a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new/empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    pop = json.loads((repo/'analysis/figure4/task_populations.json').read_text())
    data = {}; hashes = {}; cache = None
    if a.verified_reconstruction:
        cache = json.loads(a.verified_reconstruction.read_text())
        hashes['reconstruction_cache_sha256'] = hashlib.sha256(a.verified_reconstruction.read_bytes()).hexdigest()
    for model in MODELS:
        if cache is None:
            data[model], hh = reconstruct(repo, out, model); hashes.update(hh)
        else:
            data[model] = cache[model]
    data['robotics_er2'], hh = reconstruct(repo, out, 'robotics_er2'); hashes.update(hh)
    common = sorted(set.intersection(*(set(data[m]) for m in MODELS)))
    assert len(common) == 1651
    results = {}; rows = []; memberships = {}

    def add(model, population, metric, stat, numerator=None):
        rows.append(dict(model=model, population=population, metric=metric,
                         estimate=stat.get('estimate'), ci95_low=stat.get('ci95', [None, None])[0],
                         ci95_high=stat.get('ci95', [None, None])[1], n=stat.get('n'),
                         n_tasks=stat.get('n_tasks', stat.get('n')), layouts=stat.get('layouts'),
                         numerator=numerator, bootstrap_draws=stat.get('bootstrap_draws', stat.get('valid_draws')),
                         bootstrap_seed=stat.get('bootstrap_seed')))

    for model, units in data.items():
        is_er = model == 'robotics_er2'
        source_pops = {'main': sorted(units)} if is_er else {'main': pop['main1641'], 'broad': common}
        source_pops['canonical'] = sorted(g for g in units if g.startswith('canonical_'))
        source_pops['transfer'] = pop['er2_historical125'] if is_er else pop['canonical118']
        selected = {k: [g for g in ids if units[g]['layout'] not in set(a.exclude_layouts)] for k, ids in source_pops.items()}
        memberships[model] = {k: dict(original_n=len(ids), retained_n=len(selected[k]), retained=selected[k],
                                      excluded=[g for g in ids if g not in selected[k]]) for k, ids in source_pops.items()}
        r = dict(solve_rates={}, feedback={}, interaction={}, canonical_controls={})
        for name in ['main'] + ([] if is_er else ['broad']):
            ids = selected[name]; ls = [units[g]['layout'] for g in ids]; rate = {}
            for condition in ['full', 'neither', 'trace_status']:
                label = f'fullbank-{model}-{condition}-solve' if name == 'broad' else f'audit-main-{model}-{condition}-solve'
                values = [units[g]['arms'][condition]['solved'] for g in ids]
                stat = bootstrap(values, ls, label_seed(label), 10000); stat['solved'] = int(sum(values))
                rate[condition] = stat; add(model, name, condition+'_solve_rate', stat, sum(values))
            for c1, c2 in [('full', 'neither'), ('trace_status', 'neither'), ('trace_status', 'full')]:
                values = [units[g]['arms'][c1]['solved'] - units[g]['arms'][c2]['solved'] for g in ids]
                stat = bootstrap(values, ls); rate[c1+'_minus_'+c2] = stat
                add(model, name, c1+'_minus_'+c2+'_solve', stat, sum(values))
            r['solve_rates'][name] = rate
        ids = selected['main']; ls = [units[g]['layout'] for g in ids]
        first = [units[g]['arms']['full']['attempts'][0]['S'] for g in ids]
        stat = bootstrap(first, ls, label_seed(f'audit-main-{model}-initial'), 10000)
        r['interaction']['initial_success'] = stat; add(model, 'main', 'initial_success', stat, sum(first))
        for label, fn in [('chosen_pq_initial', lambda u: u['arms']['full']['attempts'][0]['pq150']),
                          ('chosen_pq_final', lambda u: u['arms']['full']['attempts'][-1]['pq150']),
                          ('chosen_pq_gain', lambda u: u['arms']['full']['attempts'][-1]['pq150'] - u['arms']['full']['attempts'][0]['pq150'])]:
            stat = bootstrap([fn(units[g]) for g in ids], ls)
            r['interaction'][label] = stat; add(model, 'main', label, stat)
        failed = [g for g in ids if not units[g]['arms']['full']['attempts'][0]['S']]
        early = [g for g in selected['transfer'] if not units[g]['arms']['full']['attempts'][0]['S']]
        for g in ids:
            for key in ['S', 'coords', 'pq150']:
                assert units[g]['arms']['full']['attempts'][0][key] == units[g]['arms']['neither']['attempts'][0][key]
        for label, gg, fn in [
            ('subsequent_solve', failed, lambda u: u['arms']['full']['solved']-u['arms']['neither']['solved']),
            ('next_chosen_pq', failed, lambda u: u['arms']['full']['attempts'][1]['pq150']-u['arms']['neither']['attempts'][1]['pq150']),
            ('early_novel_pq', early, lambda u: u['transfer']['full']['early_post']['pq150']-u['transfer']['neither']['early_post']['pq150'])]:
            stat = bootstrap([fn(units[g]) for g in gg], [units[g]['layout'] for g in gg])
            r['feedback'][label] = stat; add(model, 'shared_first_failure_'+('transfer' if label=='early_novel_pq' else 'main'), label, stat)
        ids = selected['transfer']; ls = [units[g]['layout'] for g in ids]
        for label, fn in [('novel_pre', lambda u: u['transfer']['full']['terminal_pre']['pq150']),
                          ('novel_post', lambda u: u['transfer']['full']['terminal_post']['pq150']),
                          ('novel_gain', lambda u: u['transfer']['full']['terminal_post']['pq150']-u['transfer']['full']['terminal_pre']['pq150']),
                          ('novel_gain_beyond_retry', lambda u: (u['transfer']['full']['terminal_post']['pq150']-u['transfer']['full']['terminal_pre']['pq150'])-(u['transfer']['neither']['terminal_post']['pq150']-u['transfer']['neither']['terminal_pre']['pq150']))]:
            stat = bootstrap([fn(units[g]) for g in ids], ls)
            r['interaction'][label] = stat; add(model, 'transfer', label, stat)
        ids = selected['canonical']; ls = [units[g]['layout'] for g in ids]
        cc = [c for c in ['full', 'neither', 'trace_status', 'status_only', 'frames_only'] if all(c in units[g]['arms'] for g in ids)]
        for c in cc:
            vv = [units[g]['arms'][c]['solved'] for g in ids]; stat = bootstrap(vv, ls, 6112601, 5000)
            r['canonical_controls'][c] = stat; add(model, 'canonical', c+'_solve_rate', stat, sum(vv))
        for c1,c2 in [('full','status_only'),('status_only','neither'),('frames_only','neither'),('full','frames_only')]:
            if c1 in cc and c2 in cc:
                vv = [units[g]['arms'][c1]['solved']-units[g]['arms'][c2]['solved'] for g in ids]
                stat = bootstrap(vv, ls, 6112601, 5000); r['canonical_controls'][c1+'_minus_'+c2] = stat
                add(model, 'canonical', c1+'_minus_'+c2, stat, sum(vv))
        at, ep = observations(units, selected['main'], selected['transfer'])
        f = dict(A=est.regression(at))
        f['A']['bootstrap_seed'] = 6112702
        if not is_er:
            f.update(B=est.partial(ep, ['upward']),
                     B_richer=est.partial(ep, ['upward','P_first','N_pre','exposure']),
                     C=est.reg(ep, 'N_early','S_first',['early_pre','upward'],'layout'))
            for key in ['B','B_richer']: f[key]['bootstrap_seed'] = 6112703
            f['C']['bootstrap_seed'] = 6112702
        for key, st in f.items(): add(model, 'main' if key=='A' else 'transfer', 'figure4_'+key, st)
        f['conditional_A'] = {str(s): conditional_mean(at,s) for s in (0,1)}
        f['conditional_C'] = {str(s): est.simple([e for e in ep if e['S_first']==s],lambda rr: float(np.mean([x['N_early'] for x in rr]))) for s in (0,1)}
        r['figure4'] = f; results[model] = r
        print(model, 'main', len(selected['main']), 'transfer',len(selected['transfer']), 'done', flush=True)
    (out/'results.json').write_text(json.dumps(results, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')
    (out/'populations.json').write_text(json.dumps(memberships, indent=2)+'\n')
    (out/'input_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    with (out/'estimates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (out/'method.json').write_text(json.dumps(dict(excluded_layouts=sorted(a.exclude_layouts), seed=2026,
        analysis='Post hoc sensitivity; archived records and original estimates remain unchanged',
        intervals='Percentile whole-layout cluster bootstrap, resampling both tool directions and all goals together',
        figure4='Same frozen estimator functions, adjustment sets and seeds as the November 14 audit',
        prediction_score='PQ sigma=150 pixels; unchanged exit-state handling',
        no_human_data=True, no_inference=True),indent=2)+'\n')


if __name__ == '__main__':
    main()
