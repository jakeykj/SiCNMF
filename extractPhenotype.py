import sys
sys.path.append('/home/joyceho/git/tensor')
from extractResults import ExtractResults
import extractResults
from graniteBCD import GraniteBCD
import tensorIO
import json
import cPickle as pickle
#from hetroCNMFEstimator_B import FUb,prepFactor
import numpy as np
model='SiCNMF'
rk=20

eta=25.0
i=2

filename = '/home/suriyag/collective-mf/SiCNMF/results/0905_NO_ALPHA/vandy_SiCNMF_eta%s_i%d_rk20.pickle' %(eta,i)

X, axisDict, classDict = tensorIO.load_tensor("/home/joyceho/git/vandy/data/codeMedTen_{0}.dat")
#phewasDict = json.load(open("/home/joyceho/git/data-preprocess/data/phewas.json", "r"))
phewasDict = json.load(open("phewas.json", "r"))
ra = extractResults.reverseAxis(axisDict)
ra_phewas = {k: phewasDict[v] for k, v in ra[1].iteritems()}
ra[1] = ra_phewas

#X=np.load('data/codeMedMat_1.npy').ravel()[0].tocsc()
#loss='sparse_poisson'


def truncateFactors(U,V):    
    for j in range(U.shape[1]):
        ind=np.argsort(U[:,j])
        U[:,j][ind[:-5]]=0
        ind=np.argsort(V[:,j])
        V[:,j][ind[:-5]]=0
       
    U=np.apply_along_axis(lambda u:u/u.sum(),0,U)
    V=np.apply_along_axis(lambda u:u/u.sum(),0,V)
    return U,V

def write_marble_excel(filename, output):
    decomp = pickle.load(open(filename,'rb'))
    Ucode=decomp['Ubs'][1][:,:rk]
    Umed=decomp['Ubs'][2][:,:rk]
    Upat=decomp['Ubs'][0][:,:rk]
    S= np.sum(Upat,0)*(np.sum(Umed,0)+np.sum(Ucode,0))
    Ucode,Umed=truncateFactors(Ucode,Umed)
    Upat=np.apply_along_axis(lambda u:u/u.sum(),0,Upat)
         
    U = [Upat, Ucode, Umed]

    er = ExtractResults(U, S, ra, 0)
    er.revAxis = ra
    er.write_excel(output)

if __name__ == '__main__':
    write_marble_excel("%s" %filename, "results/%s-%s-%d.xlsx" %(model,eta,i))
