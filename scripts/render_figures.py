#!/usr/bin/env python3
"""Render vector Figures 3/4 only from the finalized numerical results."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
MODELS=['gpt5','gemini31pro','qwen36plus'];NAMES=['GPT-5','Gemini 3.1 Pro','Qwen 3.6 Plus']
COLORS=['#0072B2','#B85C00','#7B3294'];MARKERS=['s','^','D'];STYLES=['-','--',':']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9aa5ae','axes.linewidth':.65,'text.color':'#202428','axes.labelcolor':'#202428','savefig.facecolor':'white'})
def grid(ax,axis='y'):
 ax.grid(axis=axis,color='#e4e9ed',lw=.55);ax.set_axisbelow(True)
def error(ax,x,s,i,horizontal=False,open_marker=False):
 val=s['estimate'];lo,hi=s['ci95'];assert np.isfinite([val,lo,hi]).all()and lo<=hi
 mid=(lo+hi)/2;extent=(hi-lo)/2
 kw=dict(fmt='none',color=COLORS[i],elinewidth=1,capsize=2.5,zorder=4)
 if horizontal:ax.errorbar(mid,x,xerr=extent,**kw);px,py=val,x
 else:ax.errorbar(x,mid,yerr=extent,**kw);px,py=x,val
 ax.plot(px,py,marker=MARKERS[i],ls='',color=COLORS[i],mfc='white'if open_marker else COLORS[i],mec=COLORS[i],ms=4.5,mew=1,zorder=5)
def figure3(R,out):
 fig,axes=plt.subplots(2,3,figsize=(10.5,5.25));fig.subplots_adjust(left=.065,right=.99,top=.86,bottom=.13,wspace=.32,hspace=.85)
 fig.text(.065,.965,'A  Change in performance during interaction',weight='bold',fontsize=12,va='top')
 fig.legend([Line2D([],[],marker='o',mfc='white',mec='#343a40',ls=''),Line2D([],[],marker='o',color='#343a40',ls='')],['Initial','Final'],loc='upper right',bbox_to_anchor=(.99,.983),ncol=2,frameon=False,fontsize=9)
 labels=['Action success','Chosen-action prediction','Novel-action prediction'];ledger={'A':{},'B':{}}
 for j in range(3):
  ax=axes[0,j];grid(ax);ax.set_title(labels[j],fontsize=10.5,pad=8);ax.set(ylim=(-.025,1.05),yticks=[0,.25,.5,.75,1],xticks=range(3),xticklabels=NAMES)
  ax.set_xlim(-.45,2.45);ax.tick_params(axis='x',labelsize=8.2);ax.set_ylabel('Success probability'if j==0 else'Prediction quality')
  for i,m in enumerate(MODELS):
   r=R[m];ini=r['interaction'][['initial_success','chosen_pq_initial','novel_pre'][j]];fin=r['solve_rates']['main']['full']if j==0 else r['interaction'][['','chosen_pq_final','novel_post'][j]]
   ax.plot([i-.14,i+.14],[ini['estimate'],fin['estimate']],color=COLORS[i],lw=1.6);error(ax,i-.14,ini,i,open_marker=True);error(ax,i+.14,fin,i)
   ledger['A'][m+'_'+str(j)]={'initial':ini,'final':fin}
 fig.text(.065,.478,'B  Benefit of retrying with feedback versus without feedback',weight='bold',fontsize=12,va='top')
 for j,(metric,label)in enumerate(zip(['subsequent_solve','next_chosen_pq','early_novel_pq'],['Subsequent action success','Next chosen-action prediction','Novel prediction after one rollout'])):
  ax=axes[1,j];grid(ax);ax.axhline(0,color='#64737e',ls='--',lw=.8);ax.set_title(label,fontsize=10,pad=8);ax.set(xticks=range(3),xticklabels=NAMES);ax.tick_params(axis='x',labelsize=8.2);ax.set_ylabel('Difference in success'if j==0 else'Difference in prediction quality')
  for i,m in enumerate(MODELS):error(ax,i,R[m]['feedback'][metric],i);ledger['B'][m+'_'+metric]=R[m]['feedback'][metric]
  ax.set_xlim(-.45,2.45)
  ax.set_ylim(min(-.03,min(R[m]['feedback'][metric]['ci95'][0]for m in MODELS)-.02),max(R[m]['feedback'][metric]['ci95'][1]for m in MODELS)+.035)
 save(fig,out,'figure3_interaction');return ledger

def figure4(R,OBS,out):
 fig=plt.figure(figsize=(10.5,5.65));rng=np.random.default_rng(6112809);ledger={}
 for j,k in enumerate(['A','B','C']):
  left=.062+j*.326
  title={'A':'A  Action success ↔\nChosen-action prediction','B':'B  Chosen-action prediction ↔\nNovel-action prediction','C':'C  Action success ↔\nNovel-action prediction'}[k]
  fig.text(left,.975,title,weight='bold',fontsize=11,va='top',linespacing=1.15)
  fig.text(left,.865,{'A':'Individual attempts','B':'Terminal checkpoint','C':'After one rollout'}[k],fontsize=9)
  ax=fig.add_axes([left,.51,.265,.32]);grid(ax,'both');panel={'effects':{},'means':{},'descriptive_fits':{}}
  for i,m in enumerate(MODELS):
   f=R[m]['figure4'];panel['effects'][m]={k:f[k]}
   if k=='B':
    ep=OBS['episodes'][m];x=np.array([r['P_final']for r in ep]);y=np.array([r['N']for r in ep]);ax.scatter(x,y,s=10,c=COLORS[i],marker=MARKERS[i],alpha=.3,edgecolors='none');slope,intercept=np.polyfit(x,y,1);xx=np.linspace(x.min(),x.max(),50);ax.plot(xx,intercept+slope*xx,c=COLORS[i],ls=STYLES[i],lw=1.2);panel['effects'][m]['B_richer']=f['B_richer'];panel['descriptive_fits'][m]={'slope':float(slope),'intercept':float(intercept),'n':len(ep),'x_range':[float(x.min()),float(x.max())]}
   else:
    rows=OBS['attempts'][m]if k=='A'else OBS['episodes'][m];sk='S'if k=='A'else'S_first';yk='pq'if k=='A'else'N_early'
    for s in [0,1]:
     vals=[r[yk]for r in rows if r[sk]==s];x=s+(i-1)*.1;alpha=min(.2,45/max(1,len(vals)))if k=='A'else .2
     ax.scatter(x+rng.uniform(-.027,.027,len(vals)),vals,s=5,c=COLORS[i],marker=MARKERS[i],alpha=alpha,edgecolors='none',rasterized=False)
     stat=f['conditional_'+k][str(s)];error(ax,x,stat,i);panel['means'][m+'_'+str(s)]=stat
  ax.set(ylim=(-.025,1.04),yticks=[0,.5,1]);ax.set_ylabel('Chosen prediction quality'if k=='A'else'Novel prediction quality',fontsize=8.5)
  if k=='B':ax.set(xlim=(-.025,1.025),xticks=[0,.5,1]);ax.set_xlabel('Chosen prediction quality')
  else:ax.set(xlim=(-.24,1.24),xticks=[0,1],xticklabels=['Failed','Solved']);ax.set_xlabel('This attempt'if k=='A'else'First attempt')
  fig.text(left,.39,{'A':'Task + attempt controls','B':'Shared-difficulty controls','C':'Equal feedback exposure'}[k],weight='bold',fontsize=10)
  fx=fig.add_axes([left+.072,.18,.193,.17]);grid(fx,'x');fx.axvline(0,c='#64737e',ls='--',lw=.8)
  for i,m in enumerate(MODELS):
   error(fx,2-i+(.08 if k=='B'else 0),R[m]['figure4'][k],i,horizontal=True)
   if k=='B':error(fx,2-i-.12,R[m]['figure4']['B_richer'],i,horizontal=True,open_marker=True)
  fx.set(yticks=[2,1,0],yticklabels=NAMES,ylim=(-.5,2.5));fx.tick_params(axis='y',length=0,pad=4,labelsize=7.3);fx.spines['left'].set_visible(False)
  for label,color in zip(fx.get_yticklabels(),COLORS):label.set_color(color)
  fx.set_xlabel('Partial rank correlation'if k=='B'else'Difference in prediction quality',fontsize=7.8)
  low=min(R[m]['figure4'][k]['ci95'][0]for m in MODELS);high=max(R[m]['figure4'][k]['ci95'][1]for m in MODELS)
  if k=='B':low=min(low,*(R[m]['figure4']['B_richer']['ci95'][0]for m in MODELS));high=max(high,*(R[m]['figure4']['B_richer']['ci95'][1]for m in MODELS))
  span=max(.1,high-low);fx.set_xlim(min(0,low)-span*.12,max(0,high)+span*.12)
  note={'A':'Equal weight per episode','B':'Filled: layout + direction; Open: + baseline + exposure','C':'Baseline probe + layout + direction'}[k]
  fig.text(left+(.13 if k=='B'else 0),.087,note,fontsize=6.8 if k=='B'else 7.2,ha='center'if k=='B'else'left');ledger[k]=panel
 fig.legend([Line2D([],[],color=c,marker=m,ls=s,lw=1.3)for c,m,s in zip(COLORS,MARKERS,STYLES)],NAMES,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False,fontsize=9)
 save(fig,out,'figure4_correspondence');return ledger

def save(fig,out,name):
 for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=350,metadata={'Creator':'vp-2026'}if ext=='pdf'else None)
 svg=out/(name+'.svg');svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 R=json.loads((a.results/'summary.json').read_text());OBS=json.loads((a.results/'observations.json').read_text())
 ledger={'figure3':figure3(R,a.output),'figure4':figure4(R,OBS,a.output),'human_estimates':'None included; measured human results remain separate.'}
 ledger['source_sha256']={name:hashlib.sha256((a.results/name).read_bytes()).hexdigest()for name in ['summary.json','observations.json']}
 (a.output/'plotted_values.json').write_text(json.dumps(ledger,indent=2)+'\n')
if __name__=='__main__':main()
