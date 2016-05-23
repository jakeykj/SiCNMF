import seaborn as sns
import cPickle as pickle
import numpy as np
from pandas import DataFrame
import scipy.linalg as la

sns.set_style("white")
sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 1})


rk = 20
N =[2039,936,161]

niter = 5
etas = [25,50,100,500,1000,5000,10000,50000]
#etas = [0.2,0.4,0.6,0.8,1.0]
#etas = [25,50]
nvals = len(etas)
sid =range(20)

etas = etas[:nvals]
sid = sid[:nvals]


f = np.zeros((len(etas),niter))
nit = np.zeros((len(etas),niter))
t = np.zeros((len(etas),niter))
x = {'eta':[],'nnzMed':[],'nnzCode':[],'nnzPat':[],'run':[],'sid':[]};
xbest = {'eta':[],'nnzMed':[],'nnzCode':[],'nnzPat':[],'sid':[]};
best_run=np.zeros(len(etas),dtype=int)

jitter=np.arange(-0.15,0.25,0.4/niter)

model = '/home/suriyag/collective-mf/SiCNMF/results/0905_NO_ALPHA/vandy_SiCNMF_eta%s_i%d_rk20.pickle'

for ix,eta in enumerate(etas):
    for i in range(niter):
        data = pickle.load(open(model %(float(eta),i),'rb'))
        print data['f'],la.norm(data['Ubs'][0][:,:-1]),data['stat']['niter'],eta
        for k in range(len(data['Ubs'])):
            data['Ubs'][k]=data['Ubs'][k]/(data['Ubs'][k].sum(0))
        x['nnzPat'] = x['nnzPat']+[len(np.where(data['Ubs'][0][:,j]>1e-10)[0]) for j in range(rk)]
        x['nnzCode'] = x['nnzCode']+[len(np.where(data['Ubs'][1][:,j]>1e-10)[0]) for j in range(rk)]
        x['nnzMed'] = x['nnzMed']+[len(np.where(data['Ubs'][2][:,j]>1e-10)[0]) for j in range(rk)] 
        x['eta'] = x['eta']+[eta]*rk
        x['sid'] = x['sid']+[sid[ix]+jitter[i]]*rk
        x['run'] = x['run']+[i]*rk        
        f[ix,i] = data['f'].sum()
        nit[ix,i] = data['stat']['niter']
        t[ix,i] = data['t']            
        
    best_run[ix]=int(np.argmin(f[ix,:]))
    data = pickle.load(open(model %(float(eta),best_run[ix]),'rb'))    
    for k in range(len(data['Ubs'])):
        data['Ubs'][k]=data['Ubs'][k]/(data['Ubs'][k].sum(0))
    xbest['nnzPat'] = xbest['nnzPat']+[len(np.where(data['Ubs'][0][:,j]>1e-10)[0]) for j in range(rk)]
    xbest['nnzCode'] = xbest['nnzCode']+[len(np.where(data['Ubs'][1][:,j]>1e-10)[0]) for j in range(rk)]
    xbest['nnzMed'] = xbest['nnzMed']+[len(np.where(data['Ubs'][2][:,j]>1e-10)[0]) for j in range(rk)]
    xbest['eta'] = xbest['eta']+[eta]*rk
    xbest['sid'] = xbest['sid']+[sid[ix]]*rk
    
    print "Eta:%s, Best run:%d, fbest:%f, median_nnz:%f,%f" %(eta,best_run[ix],f[ix,best_run[ix]],np.median([len(np.where(data['Ubs'][1][:,j]>=1e-10)[0]) for j in range(rk)]), np.median([len(np.where(data['Ubs'][2][:,j]>=1e-10)[0]) for j in range(rk)]))

    
# Plotting
x = DataFrame(data=x)
xbest = DataFrame(data=xbest)
f = DataFrame(data=f)
t = DataFrame(data=t)
label=list(['%0.5g' %etas[i] for i in range(nvals)])
sns.set_palette(sns.color_palette("dark",niter), n_colors=niter)

fig,ax=sns.plt.subplots(2,3,figsize=(25,15))

# All Runs
for i in range(niter):
    sns.regplot(x="sid",y="nnzMed",data=x[x['run']==i],fit_reg=False, ax=ax[0,0], color=sns.color_palette()[i])
    sns.regplot(x="sid",y="nnzCode",data=x[x['run']==i],fit_reg=False, ax=ax[0,1], color=sns.color_palette()[i])
    ax[0,2].plot((sid+jitter[i])[:len(f[i])],f[i],'o', color=sns.color_palette()[i])
    
# Best Run
best_palette=[sns.color_palette()[0]]*(max(sid)+1)
for ix in range(len(etas)):
    best_palette[sid[ix]]=sns.color_palette()[int(best_run[ix])]
    
sns.boxplot(x='sid',y='nnzMed',data=xbest,ax=ax[1,0],order=np.arange(max(sid)+1),\
            palette=best_palette, fliersize=0, whis=1.5); 
for ix in sid:   
    bp=sns.plt.boxplot(np.array(xbest['nnzMed'][xbest['sid']==ix]))
    q3=bp['whiskers'][1].get_ydata()[1]
    med=bp['medians'][0].get_ydata()[1]
    ax[1,0].text(ix, q3+1, '%1.2f\n(%1.2f)'%(med,q3), size=15, rotation=90, ha="center", va="bottom")

sns.boxplot(x='sid',y='nnzCode',data=xbest,ax=ax[1,1],order=np.arange(max(sid)+1),\
            palette=best_palette, fliersize=0, whis=1.5); 
for ix in sid:   
    bp=sns.plt.boxplot(np.array(xbest['nnzCode'][xbest['sid']==ix]))
    q3=bp['whiskers'][1].get_ydata()[1]
    med=bp['medians'][0].get_ydata()[1]
    ax[1,1].text(ix, q3+10, '%1.2f\n(%1.2f)'%(med,q3), size=15, rotation=90, ha="center", va="bottom")

    
    
ax[1,2].plot(sid,[f[best_run[s]][s] for s in range(len(etas))],'o-',color=sns.xkcd_rgb["pale red"])


#Axis Properties
import string
alphabet = list(string.ascii_lowercase)
for i,a in enumerate(fig.axes):
    a.set(xlabel='Sparsity Parameter $\eta$',xticklabels=label,\
          xticks=sid,xlim=[min(sid)-0.5,max(sid)+0.5])
    sns.plt.setp(a.get_xticklabels(), rotation=90)

for i in range(2):
    ax[i,0].set(ylim=(0,None),ylabel='Medication Sparsity'); 
    ax[i,1].set(ylim=(0,None),ylabel='ICD9 Sparsity'); 
    ax[i,2].set(ylabel='Divergence $\sum_v D(X_v,UA_v)$')

#,yticklabels=list(['%0.2g' %ff for ff in ax[1,0].get_yticks()]))
#ax[2,0].set(ylim=(0,None),ylabel='# Anchors'); 
#ax[3,0].set(ylabel='Time to Fit')

#ax[0,1].set(ylim=(0,None),ylabel='# Anchors'); 
#ax[1,1].set(ylim=(0,None),ylabel='Sparsity');
#ax[2,1].set(ylabel='Mean Divergence')
#ax[3,1].set(ylabel='Time to Fit')
ax[1,2].set(ylim=ax[0,2].get_ylim(),yticklabels=ax[0,2].get_yticklabels())
#if not baseline: ax[0,1].set(ylim=(0,750),ylabel='Phenotype Sparsity'); 
fig.tight_layout()
fig.savefig("SiCNMF_sparsity.pdf", dpi=300)
