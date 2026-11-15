"""Frozen estimator helpers retained from the independently verified audit."""
import collections,hashlib
import numpy as np

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

def observations(units, broad, canonical, condition='full', sigma=150):
    attempts=[];episodes=[];key=f'pq{sigma}'
    for gid in broad:
        u=units[gid];aa=u['arms'][condition]['attempts']
        assert len(aa)==u['arms'][condition]['count']
        assert bool(max(a['S'] for a in aa))==bool(u['arms'][condition]['solved'])
        for a in aa:
            attempts.append(dict(goal=gid,layout=u['layout'],attempt=a['attempt'],S=a['S'],pq=a[key],episode_n=len(aa)))
    for gid in canonical:
        u=units[gid];aa=u['arms'][condition]['attempts'];t=u['transfer'][condition]
        assert np.linalg.norm(np.array(u['early_probe_coords'])-aa[0]['coords'])>=40
        episodes.append(dict(goal=gid,layout=u['layout'],upward=u['upward'],S_first=aa[0]['S'],
            P_first=aa[0][key],P_final=aa[-1][key],N_pre=t['terminal_pre'][key],N=t['terminal_post'][key],
            exposure=len(aa),early_pre=t['early_pre'][key],early_post=t['early_post'][key],N_early=t['early_post'][key]))
    return attempts,episodes

def conditional_mean(rows,success):
    rr=[r for r in rows if r['S']==success];labs=sorted({r['layout']for r in rr})
    num=np.array([sum(r['pq']/r['episode_n']for r in rr if r['layout']==l)for l in labs])
    den=np.array([sum(1/r['episode_n']for r in rr if r['layout']==l)for l in labs])
    sample=np.random.default_rng(6112701).integers(len(labs),size=(2000,len(labs)))
    bs=num[sample].sum(axis=1)/den[sample].sum(axis=1)
    return dict(estimate=float(num.sum()/den.sum()),ci95=np.quantile(bs,[.025,.975]).tolist(),n=len(rr))
