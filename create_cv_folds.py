import sys
sys.path.append('/home/joyceho/git/tensor')
import tensorIO
import pandas as pd
import json
import numpy as np

X, axisDict, classDict = tensorIO.load_tensor("/home/joyceho/git/vandy/data/codeMedTen_{0}.dat")

patDict = axisDict[0]
nPat=len(patDict)

cohortClass=json.load(open('/home/joyceho/git/vandy/phenotype/cohort-classes.json','r'))

patClass = {k: cohortClass[k] for k in patDict.keys()}

classDF = pd.DataFrame(patClass.items(), columns=['ID', 'Class'])


assert ([x for x in patDict.values() if x not in range(nPat)] == []), "Patient dict values is not 0:nPat"

class_labels=np.zeros((nPat,2))
for key in patDict.keys():
    if patClass[key]=='ctrl':
        class_labels[patDict[key]]=[0,0]
    elif patClass[key]=='t2d':
        class_labels[patDict[key]]=[1,0]
    elif patClass[key]=='res_htn':
        class_labels[patDict[key]]=[0,1]
    elif patClass[key]=='all':
        class_labels[patDict[key]]=[1,1]
    else: 
        assert 0, "patClass has unidentified class type %s" %patClass[key]

np.save('/home/suriyag/collective-mf/data/class_labels.npy',class_labels)

from sklearn.cross_validation import StratifiedKFold
import os
import cPickle as pickle

skf = StratifiedKFold(classDF['Class'], n_folds=5)
cv_folds_patient_indices = []
if not os.path.exists('../data/cv_folds'): os.mkdir('../data/cv_folds')

for train_index, test_index in skf:
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    cv_folds_patient_indices.append({'train':train_index,'test':test_index})
    
    X=np.load('/home/suriyag/collective-mf/data/codeMat_1.npy').ravel()[0].tocsr()
    Xs_train=[X[train_index,:].tocoo()]
    Xs_test=[X[test_index,:].tocoo()]
    
    X=np.load('/home/suriyag/collective-mf/data/medMat_1.npy').ravel()[0].tocsr()    
    Xs_train.append(X[train_index,:].tocoo())
    Xs_test.append(X[test_index,:].tocoo())
    
    pickle.dump({'Xs_train':Xs_train, 'Xs_test':Xs_test, \
                'class_train':class_labels[train_index,:], 'class_test':class_labels[test_index,:]},\
                open('/home/suriyag/collective-mf/data/cv_folds/data_cv%d.pkl' %(len(cv_folds_patient_indices)-1), 'wb'))
    
np.save('/home/suriyag/collective-mf/data/cv_fold_patient_indices.npy',cv_folds_patient_indices)     
    