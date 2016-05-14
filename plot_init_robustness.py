"""
@author: suriya
"""
import cPickle as pickle
import numpy as np
from pandas import DataFrame
import seaborn as sns
import numpy.linalg as la 
from munkres import Munkres
from scipy.spatial.distance import cosine, jaccard, hamming
sns.set_style("whitegrid")


plotmean=0

rk = 20
N =[2039,936,161]
niter = 5
etas = [25,50]#,100,500,1000,5000,10000,50000]
nvals = len(etas)
sid =[1,2,3,4,5,6,7,8,9,10]

etas = etas[:nvals]
sid = sid[:nvals]

c={'eta':[],'sid':[],'id':[],'med_cosine':[],'med_hamming':[],'med_jaccard_topK':[], 'med_hamming_topK':[],'code_cosine':[],'code_hamming':[],'code_jaccard_topK':[], 'code_hamming_topK':[],'r':[]}
cmean={'eta':[],'sid':[],'id':[],'med_cosine':[],'med_hamming':[],'med_jaccard_topK':[], 'med_hamming_topK':[],'code_cosine':[],'code_hamming':[],'code_jaccard_topK':[], 'code_hamming_topK':[]}


def fms(Wi,Wj,Ai,Aj,Si,Sj):
    cosine_fact_prod=[np.ones((rk,rk)) for v in range(V)]
    for r in range(rk):
        assert (la.norm(Wi[:,r])-1.0)<1e-15, la.norm(Wi[:,r])
        assert (la.norm(Wj[:,r])-1.0)<1e-15, la.norm(Wj[:,r])
        assert (la.norm(Ai[0][:,r])-1.0)<1e-15, la.norm(Ai[0][:,r])
        assert (la.norm(Ai[1][:,r])-1.0)<1e-15, la.norm(Ai[1][:,r])
        assert (la.norm(Aj[0][:,r])-1.0)<1e-15, la.norm(Aj[0][:,r])
        assert (la.norm(Aj[1][:,r])-1.0)<1e-15, la.norm(Aj[1][:,r])
        
    for v in range(V):        
        cosine_fact_prod[v]=np.multiply(np.dot(Wi.T,Wj),np.dot(Ai[v].T,Aj[v]))
    
    S1=[np.tile(Si[v],(rk,1)).T for v in range(V)]
    S2=[np.tile(Sj[v],(rk,1)) for v in range(V)]
    weight = [np.abs(S1[v]-S2[v])/np.maximum(S1[v],S2[v]) for v in range(V)]
    return np.sum((np.ones((rk, rk)) - weight[v])*cosine_fact_prod[v] for v in range(V))


def cosine_corr(Ai,Aj):
    cosine_fact_prod=[np.ones((rk,rk)) for v in range(V)]
    for r in range(rk):
        assert (la.norm(Ai[0][:,r])-1.0)<1e-15, la.norm(Ai[0][:,r])
        assert (la.norm(Ai[1][:,r])-1.0)<1e-15, la.norm(Ai[1][:,r])
        assert (la.norm(Aj[0][:,r])-1.0)<1e-15, la.norm(Aj[0][:,r])
        assert (la.norm(Aj[1][:,r])-1.0)<1e-15, la.norm(Aj[1][:,r])        
    for v in range(V):        
        cosine_fact_prod[v]=np.dot(Ai[v].T,Aj[v])    
    return np.sum(cosine_fact_prod[v] for v in range(V))
    
def opt_index(C):
    copyC = np.ones(C.shape) - C.copy()
    hAlg = Munkres()
    indexes = hAlg.compute(copyC)
    rowIdx, colIdx = map(np.array, zip(*indexes))
    return rowIdx, colIdx, indexes


model = '/home/suriyag/collective-mf/SiCNMF/results/vandy_SiCNMF_eta%s_i%d_rk20.pickle'
jitter=np.arange(-0.15,0.25,0.4/niter)
V=2
K=10 # Different from V+1, K of topK 
for ix,eta in enumerate(etas):
    print eta
    ids=0
    W=[[]]*niter
    A=[[]]*niter
    S=[[]]*niter
    Stemp=[[]]*niter
    for i in range(niter):
        data = pickle.load(open(model %(float(eta),i),'rb'))        
        W[i]=data['Ubs'][0][:,:rk]; 
        A[i]=[[]]*V;S[i]=[[]]*V
        for v in range(V):
            A[i][v] = data['Ubs'][v+1][:,:rk];   
            S[i][v] = np.zeros(rk)
            for r in range(rk):
                S[i][v][r] = la.norm(W[i][:,r])*la.norm(A[i][v][:,r])                
                A[i][v][:,r] = A[i][v][:,r]/la.norm(A[i][v][:,r])
                
            
        for r in range(rk):
            W[i][:,r] = W[i][:,r]/la.norm(W[i][:,r]) 

        
    for i in range(niter):        
        Acodei = np.copy(A[i][0])
        Amedi = np.copy(A[i][1])
        for j in range(i+1,niter):            
            #C =  fms(W[i],W[j],A[i],A[j],S[i],S[j])
            C =  cosine_corr(A[i],A[j])
            rowIx,colIx,idx = opt_index(C)
   
            Acodej = A[j][0][:,colIx]
            Amedj = A[j][1][:,colIx]
            #cmean['fms']=cmean['fms']+np.mean([C[k,colIx[k]] for k in range(rk)])
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
#ax[0,0].legend(handles, labels, loc="lower left")    
#ax[0,0].set(ylabel="Medication Cosine")
#ax[0,1].set(ylabel="Medication Hamming on Support")
#ax[0,2].set(ylabel="Medication Intersect on Top %d" %K)
#ax[1,0].set(ylabel="ICD9 Cosine")
#ax[1,1].set(ylabel="ICD9 Hamming on Support")
#ax[1,2].set(ylabel="ICD9 Intersect on Top %d" %K)


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
#ax[0,0].legend(handles, labels, loc="lower left")    
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
