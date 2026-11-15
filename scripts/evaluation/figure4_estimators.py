"""Frozen original model estimators; copied function definitions only, no human loaders."""

import numpy as np

from scipy.stats import rankdata

DRAW=2000

def regression(rows):
 names=['S']+[f'attempt_{k}'for k in range(2,9)]
 X=np.array([[r['S']]+[int(r['attempt']==k)for k in range(2,9)]for r in rows],float);Y=np.array([r['pq']for r in rows]);w=np.array([1/r['episode_n']for r in rows])
 goals=sorted({r['goal']for r in rows});layout=sorted({r['layout']for r in rows});ga=np.array([r['goal']for r in rows]);la=np.array([r['layout']for r in rows])
 for g in goals:
  ix=np.flatnonzero(ga==g);assert np.ptp(w[ix])==0;X[ix]-=X[ix].mean(axis=0);Y[ix]-=Y[ix].mean()
 keep=np.std(X,axis=0)>1e-11;X=X[:,keep];names=[n for n,k in zip(names,keep)if k];X*=np.sqrt(w[:,None]);Y*=np.sqrt(w)
 xx=[];xy=[]
 for l in layout:
  ix=np.flatnonzero(la==l);xx.append(X[ix].T@X[ix]);xy.append(X[ix].T@Y[ix])
 xx=np.array(xx);xy=np.array(xy);point=np.linalg.solve(xx.sum(axis=0),xy.sum(axis=0))[0]
 assert abs(point-np.linalg.lstsq(X,Y,rcond=None)[0][0])<1e-10
 rng=np.random.default_rng(6112702);sample=rng.integers(len(layout),size=(DRAW,len(layout)));weights=np.array([np.bincount(s,minlength=len(layout))for s in sample])
 bxx=np.einsum('dl,lij->dij',weights,xx);bxy=np.einsum('dl,li->di',weights,xy)
 valid=np.linalg.matrix_rank(bxx)==X.shape[1];bs=np.linalg.solve(bxx[valid],bxy[valid,:,None])[:,0,0]
 recovered=sum(any(r['S']==1 for r in rows if r['goal']==g)and any(r['S']==0 for r in rows if r['goal']==g)for g in goals)
 return dict(estimate=float(point),ci95=np.quantile(bs,[.025,.975]).tolist(),n=len(rows),n_tasks=len(goals),layouts=len(layout),valid_draws=int(valid.sum()),bootstrap_draws=DRAW,fixed_effect='goal',fixed_effect_levels=len(goals),parameters=names,weighting='equal episode',formula='pq ~ success + categorical attempt + exact-goal FE',informative_recovered_goals=recovered,interpretation='Conditional within-goal association, not a causal effect.')

def groups(ls):
 return [np.flatnonzero(np.asarray(ls)==k)for k in sorted(set(ls))]

def summary(est,b,n,**kw):
 b=np.asarray(b);b=b[np.isfinite(b)]
 return {'estimate':float(est),'ci95':np.quantile(b,[.025,.975]).tolist()if len(b)else None,'n':int(n),'valid_draws':int(len(b)),**kw}

def simple(rows,fn):
 gg=groups([r['layout']for r in rows]);rng=np.random.default_rng(6112701);bb=[]
 for _ in range(DRAW):
  rr=[rows[i]for k in rng.integers(len(gg),size=len(gg))for i in gg[k]]
  try:bb.append(fn(rr))
  except (ZeroDivisionError,ValueError):pass
 return summary(fn(rows),bb,len(rows),layouts=len(gg))

def demean(z,gs):
 z=z.copy()
 for ix in gs:z[ix]-=z[ix].mean(axis=0)
 return z

def reg(rows,y,target,covs=(),fe=None,equal_episode=False):
 names=[target]+list(covs);X=np.array([[r[k]for k in names]for r in rows],float);Y=np.array([r[y]for r in rows],float)
 fg=groups([r[fe]for r in rows])if fe else []
 if fe:X=demean(X,fg);Y=demean(Y[:,None],fg).ravel()
 else:X=np.column_stack([X,np.ones(len(rows))]);names+=['intercept']
 keep=np.std(X,axis=0)>1e-11
 if not fe:keep[-1]=True
 dropped=[n for n,k in zip(names,keep)if not k];names=[n for n,k in zip(names,keep)if k];X=X[:,keep]
 if target not in names:return {'estimate':None,'reason':'No within-group predictor variation'}
 at=names.index(target);w=np.array([1/r['episode_n']if equal_episode else 1 for r in rows]);X=X*np.sqrt(w[:,None]);Y=Y*np.sqrt(w)
 scale=np.sqrt((X*X).mean(axis=0));X/=scale
 rank=np.linalg.matrix_rank(X);assert rank==X.shape[1],(target,names,rank)
 beta=np.linalg.lstsq(X,Y,rcond=None)[0][at]/scale[at];gg=groups([r['layout']for r in rows]);rng=np.random.default_rng(6112702);bb=[]
 for _ in range(DRAW):
  ix=np.concatenate([gg[k]for k in rng.integers(len(gg),size=len(gg))])
  if np.linalg.matrix_rank(X[ix])==X.shape[1]:bb.append(np.linalg.lstsq(X[ix],Y[ix],rcond=None)[0][at]/scale[at])
 return summary(beta,bb,len(rows),layouts=len(gg),fixed_effect=fe,fixed_effect_levels=len(fg),parameters=names,dropped_constant_columns=dropped,residual_df=len(rows)-len(fg)-rank,weighting='equal episode'if equal_episode else'equal observation',formula=y+' ~ '+target+' + '+' + '.join(covs)+(f' + {fe} FE'if fe else''),interpretation='Conditional association, not a causal effect.')

def partial(rows,covs,fe=True):
 names=['P_final','N']+list(covs);z=np.array([[r[k]for k in names]for r in rows],float);gg=groups([r['layout']for r in rows])
 def fit(z,fg):
  q=rankdata(z,axis=0);q=demean(q,fg)if fe else q-q.mean(axis=0)
  yy=q[:,:2];xx=q[:,2:]
  if xx.shape[1]:yy-=xx@np.linalg.lstsq(xx,yy,rcond=None)[0]
  den=np.sqrt((yy[:,0]**2).sum()*(yy[:,1]**2).sum())
  return float((yy[:,0]*yy[:,1]).sum()/den)if den>1e-12 else np.nan
 est=fit(z,gg);rng=np.random.default_rng(6112703);bb=[]
 for _ in range(DRAW):
  chosen=rng.integers(len(gg),size=len(gg));ix=np.concatenate([gg[k]for k in chosen]);sizes=[len(gg[k])for k in chosen];ends=np.cumsum([0]+sizes);fg=[np.arange(ends[j],ends[j+1])for j in range(len(sizes))];bb.append(fit(z[ix],fg))
 return summary(est,bb,len(rows),layouts=len(gg),estimand='Partial Spearman: correlation of residual rank scores',controls=list(covs),layout_fixed_effect=fe,informative_paired_layouts=sum(len(g)>1 for g in gg),interpretation='Ranks and nuisance regressions are refitted in every whole-layout resample; pointwise exploratory interval.')
