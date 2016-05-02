"""
@author: suriya
"""
import cPickle as pickle
import numpy as np
from pandas import DataFrame
import seaborn as sns
import numpy.linalg as la 
from scipy.spatial.distance import cosine, jaccard, hamming
sns.set_style("whitegrid")


plotmean=0

rk = 20
N =[2039,936,161]
nvals = 7
niter = 6
etas = [1000.0,100.0,10.0,1.0,0.1,0.01,0.001]
sid =[1,2,3,4,5,6,7,8,9,10]

etas = etas[:nvals]
sid = sid[:nvals]

c={'eta':[],'sid':[],'id':[],'med_cosine':[],'med_hamming':[],'med_jaccard_topK':[], 'med_hamming_topK':[],'code_cosine':[],'code_hamming':[],'code_jaccard_topK':[], 'code_hamming_topK':[],'r':[]}
cmean={'eta':[],'sid':[],'id':[],'med_cosine':[],'med_hamming':[],'med_jaccard_topK':[], 'med_hamming_topK':[],'code_cosine':[],'code_hamming':[],'code_jaccard_topK':[], 'code_hamming_topK':[]}

    
model = '/home/suriyag/collective-mf/SiCNMF/results/3004/vandy_SiNMF_eta%s_i%d_rk20_Factors.pickle'
print model
jitter=np.arange(-0.15,0.25,0.4/niter)

K=5
for ix,eta in enumerate(etas):
    print eta
    ids=0
    for i in range(1,niter):
        data = pickle.load(open(model %(eta,i),'rb'))        
        Amedi=data['Ubs'][1][:,:rk]; 
        Acodei=data['Ubs'][2][:,:rk]; 
        for j in range(i+1,niter):            
            data = pickle.load(open(model %(eta,j),'rb'))
            Amedj=data['Ubs'][1][:,:rk]; 
            Acodej=data['Ubs'][2][:,:rk]; 
            
            # Med stats
            cc=[1-cosine(Amedi[:,r],Amedj[:,r]) for r in range(rk)]
            c['med_cosine']=c['med_cosine']+cc
            cmean['med_cosine']=cmean['med_cosine']+[np.mean(cc)]

            cc=[1-hamming(Amedi[:,r]>=1e-15,Amedj[:,r]>=1e-15) for r in range(rk)]
            c['med_hamming']=c['med_hamming']+cc
            cmean['med_hamming']=cmean['med_hamming']+[np.mean(cc)]
            
            t1=[np.argsort(Amedi[:,r])[::-1][:K] for r in range(rk)]
            t2=[np.argsort(Amedj[:,r])[::-1][:K] for r in range(rk)]
            
            cc=[len(np.intersect1d(t1[r],t2[r]))/float(len(np.union1d(t1[r],t2[r]))) for r in range(rk)]
            c['med_jaccard_topK']=c['med_jaccard_topK']+cc
            cmean['med_jaccard_topK']=cmean['med_jaccard_topK']+[np.mean(cc)]
            
            cc=[len(np.intersect1d(t1[r],t2[r]))/float(K) for r in range(rk)]
            c['med_hamming_topK']=c['med_hamming_topK']+cc
            cmean['med_hamming_topK']=cmean['med_hamming_topK']+[np.mean(cc)]
            
            # code stats
            cc=[1-cosine(Acodei[:,r],Acodej[:,r]) for r in range(rk)]
            c['code_cosine']=c['code_cosine']+cc
            cmean['code_cosine']=cmean['code_cosine']+[np.mean(cc)]

            cc=[1-hamming(Acodei[:,r]>=1e-15,Acodej[:,r]>=1e-15) for r in range(rk)]
            c['code_hamming']=c['code_hamming']+cc
            cmean['code_hamming']=cmean['code_hamming']+[np.mean(cc)]
            
            t1=[np.argsort(Acodei[:,r])[::-1][:K] for r in range(rk)]
            t2=[np.argsort(Acodej[:,r])[::-1][:K] for r in range(rk)]
            
            cc=[len(np.intersect1d(t1[r],t2[r]))/float(len(np.union1d(t1[r],t2[r]))) for r in range(rk)]
            c['code_jaccard_topK']=c['code_jaccard_topK']+cc
            cmean['code_jaccard_topK']=cmean['code_jaccard_topK']+[np.mean(cc)]
            
            cc=[len(np.intersect1d(t1[r],t2[r]))/float(K) for r in range(rk)]
            c['code_hamming_topK']=c['code_hamming_topK']+cc
            cmean['code_hamming_topK']=cmean['code_hamming_topK']+[np.mean(cc)]
            
            # Rest of stats
            c['eta']=c['eta']+[eta+jitter[i]]*rk
            cmean['eta']=cmean['eta']+[eta+jitter[i]]
            
            c['sid']=c['sid']+[sid[ix]+jitter[i]]*rk
            cmean['sid']=cmean['sid']+[sid[ix]+jitter[i]]
            
            c['id']=c['id']+[ids]*rk; 
            cmean['id']=cmean['id']+[ids];
            ids=ids+1
            
            c['r']=c['r']+range(rk)
            

# Plotting
i_ind=[]
for i in range(niter-1):
    i_ind =i_ind + [i]*(niter-1-i)

label=list(['%0.5g' %etas[i] for i in range(nvals)])
sns.set_palette(sns.color_palette("dark",niter), n_colors=niter-1)
sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 1,"font.weight":"bold","axes.labelweight":"bolder"})    

plotmean=0
c = DataFrame(data=c)
fig,ax=sns.plt.subplots(2,3, figsize=(30,20))


# All Runs
for ii in range(ids):
    sns.regplot(x="sid",y="med_cosine",data=c[c['id']==ii],fit_reg=False, ax=ax[0,0],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="med_hamming",data=c[c['id']==ii],fit_reg=False, ax=ax[0,1],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="med_hamming_topK",data=c[c['id']==ii],fit_reg=False, ax=ax[0,2],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="code_cosine",data=c[c['id']==ii],fit_reg=False, ax=ax[1,0],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="code_hamming",data=c[c['id']==ii],fit_reg=False, ax=ax[1,1],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="code_hamming_topK",data=c[c['id']==ii],fit_reg=False, ax=ax[1,2],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})

#handles=[sns.plt.plot(1, color=sns.color_palette()[i])[0] for i in range(niter-1)]
#labels=['initialization '+str(i+1) for i in range(niter-1)] 
ax[0,0].legend(handles, labels, loc="lower left")    
ax[0,0].set(ylabel="Medication Cosine")
ax[0,1].set(ylabel="Medication Hamming on Support")
ax[0,2].set(ylabel="Medication Intersect on Top %d" %K)
ax[1,0].set(ylabel="ICD9 Cosine")
ax[1,1].set(ylabel="ICD9 Hamming on Support")
ax[1,2].set(ylabel="ICD9 Intersect on Top %d" %K)


#Axis Properties
import string
for i,a in enumerate(fig.axes):
    a.set(xlabel='Regularization parameter, $\eta$',xticklabels=label,\
             xticks=sid,xlim=[min(sid)-0.5,max(sid)+0.5],ylim=(0,1.0))
    sns.plt.setp(a.get_xticklabels(), rotation=90)


fig.tight_layout()
fig.savefig('SiCNMF_robustness_initialization_'+str(plotmean)+'.pdf', dpi=300)

plotmean=1
c = DataFrame(data=cmean)
fig,ax=sns.plt.subplots(2,3, figsize=(30,20))

# All Runs
for ii in range(ids):
    sns.regplot(x="sid",y="med_cosine",data=c[c['id']==ii],fit_reg=False, ax=ax[0,0],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="med_hamming",data=c[c['id']==ii],fit_reg=False, ax=ax[0,1],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="med_hamming_topK",data=c[c['id']==ii],fit_reg=False, ax=ax[0,2],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="code_cosine",data=c[c['id']==ii],fit_reg=False, ax=ax[1,0],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="code_hamming",data=c[c['id']==ii],fit_reg=False, ax=ax[1,1],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})
    sns.regplot(x="sid",y="code_hamming_topK",data=c[c['id']==ii],fit_reg=False, ax=ax[1,2],\
                scatter_kws={'c':sns.color_palette()[i_ind[ii]]})

#handles=[sns.plt.plot(1, color=sns.color_palette()[i])[0] for i in range(niter-1)]
#labels=['initialization '+str(i+1) for i in range(niter-1)] 
ax[0,0].legend(handles, labels, loc="lower left")    
ax[0,0].set(ylabel="Medication Cosine")
ax[0,1].set(ylabel="Medication Hamming on Support")
ax[0,2].set(ylabel="Medication Intersect on Top %d" %K)
ax[1,0].set(ylabel="ICD9 Cosine")
ax[1,1].set(ylabel="ICD9 Hamming on Support")
ax[1,2].set(ylabel="ICD9 Intersect on Top %d" %K)


#Axis Properties
import string
for i,a in enumerate(fig.axes):
    a.set(xlabel='Regularization parameter, $\eta$',xticklabels=label,\
             xticks=sid,xlim=[min(sid)-0.5,max(sid)+0.5],ylim=(0,1.0))
    sns.plt.setp(a.get_xticklabels(), rotation=90)


fig.tight_layout()
fig.savefig('SiCNMF_robustness_initialization_'+str(plotmean)+'.pdf', dpi=300)
