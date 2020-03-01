import numpy as np
from scipy import linalg as LA
import sys
import scipy.io
import numpy.matlib
import copy
import scipy
import scipy.linalg
import sklearn.preprocessing
import matplotlib.pyplot as plt

def InferenceStep(Y,T,A,W,gamma,s2,b2,B,p):
    N = gamma.shape[1]
    d,M = A.shape

    if p['TD']:
        dLds = np.zeros((N,2))
    else:
        dLds = np.zeros((N,1))

    if p['DA']:
        Sigma = np.zeros((M,N))
        mu = np.zeros((M,N))
        
        if p['TD']:
            C = W.shape[0]
            AtA_s2inv = A.T.dot(A)/s2
            WtW_b2inv = W.T.dot(W)/b2
            AtAWtW = AtA_s2inv+WtW_b2inv
            AtY_s2inv = np.dot(A.transpose(),Y)/s2 + np.dot(W.transpose(),T)/b2

            J_A = np.zeros((d,d,M))
            for m in range(np.int32(M)):
                J_A[:,:,m] = np.outer(A[:,m],A[:,m])
            J_A_2d = np.reshape(J_A,(d**2,M))

            J_W = np.zeros((C,C,M))
            for m in range(np.int32(M)):
                J_W[:,:,m] = np.outer(W[:,m],W[:,m])
            J_W_2d = np.reshape(J_W,(C**2,M))
            
            for n in range(N):
                g = B.dot(gamma[:,n])
                tmp = AtAWtW+np.diag(1.0/(g))
                mu[:,n],_ = scipy.sparse.linalg.cg(tmp,AtY_s2inv[:,n])
                Sigma[:,n] = 1.0/np.diag(tmp)
                AGAt = np.diagonal(np.reshape(np.dot(J_A_2d,g),(d,d)))
                WGWt = np.diagonal(np.reshape(np.dot(J_W_2d,g),(C,C)))
                dLds[n,0] = np.sum(-np.divide(np.square(Y[:,n]),np.square(AGAt+s2))+np.reciprocal(AGAt+s2))
                dLds[n,1] = np.sum(-np.divide(np.square(T[:,n]),np.square(WGWt+b2))+np.reciprocal(WGWt+b2))
        else:
            AtA = A.T.dot(A)
            AtY_s2inv = A.T.dot(Y)/s2
            J = np.zeros((d,d,M))
            for m in range(np.int32(M)):
                J[:,:,m] = np.outer(A[:,m],A[:,m])
            J_2d = np.reshape(J,(d**2,M))

            for n in range(N):
                g = B.dot(gamma[:,n])
                tmp = AtA/s2+np.diag(1.0/(g))
                mu[:,n],_ = scipy.sparse.linalg.cg(tmp,AtY_s2inv[:,n])
                Sigma[:,n] = 1.0/np.diag(tmp)
                AGAt = np.diagonal(np.reshape(np.dot(J_2d,g),(d,d)))
                dLds[n] = np.sum(-np.divide(np.square(Y[:,n]),np.square(AGAt+s2))+np.reciprocal(AGAt+s2))
    else:
        Sigma = np.zeros((M,M,N))
        mu = np.zeros((M,N))

        if p['TD']:
            C = W.shape[0]
            
            J_A = np.zeros((d,d,M))
            for m in range(np.int32(M)):
                J_A[:,:,m] = np.outer(A[:,m],A[:,m])
            J_A_2d = np.reshape(J_A,(d**2,M))

            J_W = np.zeros((C,C,M))
            for m in range(np.int32(M)):
                J_W[:,:,m] = np.outer(W[:,m],W[:,m])
            J_W_2d = np.reshape(J_W,(C**2,M))

            J_AW = np.zeros((d,C,M))
            for m in range(np.int32(M)):
                J_AW[:,:,m] = np.outer(A[:,m],W[:,m])
            J_AW_2d = np.reshape(J_AW,(d*C,M))

            AtY_s2inv = np.dot(A.transpose(),Y)/s2+ np.dot(W.transpose(),T)/b2
            AW = np.concatenate((A,W))            
            I = LA.block_diag(s2*np.eye(d),b2*eye(C))

            for n in range(N):
                g = B.dot(gamma[:,n])
                Gamma = np.diag(g)
                AGAt = np.reshape(np.dot(J_A_2d,g),(d,d))
                WGWt = np.reshape(np.dot(J_W_2d,g),(C,C))
                AGWt = np.reshape(np.dot(J_AW_2d,g),(d,C))
                S_A,U_A = LA.eigh(AGAt,turbo=True)
                S_W,U_W = LA.eigh(WGWt,turbo=True)
                dLds[n,0] = -Y[:,n].T.dot(U_A).dot(np.diag(np.reciprocal(np.square(S_A+s2)))).dot(U_A.T).dot(Y[:,n])+np.sum(np.reciprocal(S_A+s2))
                dLds[n,1] = -T[:,n].T.dot(U_W).dot(np.diag(np.reciprocal(np.square(S_W+b2)))).dot(U_W.T).dot(T[:,n])+np.sum(np.reciprocal(S_W+b2))
                tmp = np.concatentate((AGAt,AGWt),axis=1)
                tmp = np.concatenate((tmp,np.concatenate((AGWt.T,WGWt),axis=1)))
                tmp = AW.T.dot(LA.lstsq(I+tmp,AW))
                Sigma[:,:,n] = np.subtract(Gamma,(g[:,np.newaxis]*tmp)*g)
                mu[:,n] = Sigma[:,:,n].dot(AtY_s2inv[:,n])
        else:
            J = np.zeros((d,d,M))
            AtY_s2inv = A.transpose().dot(Y)/s2
            for m in range(np.int32(M)):
                J[:,:,m] = np.outer(A[:,m],A[:,m])
            J_2d = np.reshape(J,(d**2,M))

            for n in range(N):
                g = B.dot(gamma[:,n])
                Gamma = np.diag(g)
                AGAt = np.reshape(np.dot(J_2d,g),(d,d))
                S,U = LA.eigh(AGAt,turbo=True)
                S = np.maximum(S,0.0)
                tmp = A.T.dot(U).dot(np.diag(np.reciprocal(S+s2))).dot(U.T).dot(A)
                dLds[n] = -Y[:,n].T.dot(U).dot(np.diag(np.reciprocal(np.square(S+s2)))).dot(U.T).dot(Y[:,n])+np.sum(np.reciprocal(S+s2))
                Sigma[:,:,n] = np.subtract(Gamma,(g[:,np.newaxis]*tmp)*g)
                mu[:,n] = Sigma[:,:,n].dot(AtY_s2inv[:,n])
    return Sigma,mu,dLds

def InferenceStepTree(Y,T,A,W,gamma,s2,b2,S,p):
    N = gamma.shape[1]
    d = A.shape[0]
    M = gamma.shape[0]

    if p['TD']:
        dLds = np.zeros((N,2))
    else:
        dLds = np.zeros((N,1))

    Sigma = np.zeros((M,M,N))
    mu = np.zeros((M,N))

    J = np.zeros((d,d,M))
    ASt = A.dot(S.T)

    SAtY_s2inv = S.dot(A.T).dot(Y)/s2
    for m in range(np.int32(M)):
        J[:,:,m] = np.outer(ASt[:,m],ASt[:,m])
    J_2d = np.reshape(J,(d**2,M))

    for n in range(N):
        g = gamma[:,n]
        Gamma = np.diag(g)
        AGAt = np.reshape(np.dot(J_2d,g),(d,d))
        S,U = LA.eigh(AGAt,turbo=True)
        S = np.maximum(S,0.0)
        tmp = ASt.T.dot(U).dot(np.diag(np.reciprocal(S+s2))).dot(U.T).dot(ASt)
        dLds[n] = -Y[:,n].T.dot(U).dot(np.diag(np.reciprocal(np.square(S+s2)))).dot(U.T).dot(Y[:,n])+np.sum(np.reciprocal(S+s2))
        Sigma[:,:,n] = np.subtract(Gamma,(g[:,np.newaxis]*tmp)*g)
        mu[:,n] = Sigma[:,:,n].dot(SAtY_s2inv[:,n])
    return Sigma,mu,dLds

def MultimodalSBL(Y,A,p):   
    #Y: list of datasets
    #A: list of learned dictionaries
    #p: dictionary of parameters. See MultimodalDL() for explanation of expected elements
    nS = len(Y)
    #Read in input data and perform initializations
    p['TD'] = False
    M = np.zeros(nS,dtype=int)
    s2 = np.array(np.zeros(nS))
    N = np.int(Y[0].shape[1])

    for k in range(nS):
        M[k] = A[k].shape[1]
        s2[k] = np.float(p['s2_initial'][k])
        
    #Initialize group information, including back and forward group operators
    if 'groups' in p:
        groups = copy.deepcopy(p['groups'])
    else:
        groups = []
        for k in range(nS):
            groups.append(np.array(range(M[k]))+1)
    uniqueGroups = np.array(np.squeeze(np.unique(groups[0])))

    B_back = []
    B_forward = []
    nG = np.zeros(len(uniqueGroups))
    for g in range(len(uniqueGroups)):
        for k in range(nS):
            nG[g] = nG[g]+np.float(len(np.where(groups[k] == uniqueGroups[g])[0]))

    for k in range(nS):
        B_back.append(np.zeros((np.int32(M[k]),len(uniqueGroups))))
        B_forward.append(np.zeros((len(uniqueGroups),np.int32(M[k]))))

        for g in range(len(uniqueGroups)):
            ind = np.where(groups[k] == uniqueGroups[g])[0]
            for ii in range(len(ind)):
                B_back[k][ind[ii],g] = np.float(1)
                B_forward[k][g,ind[ii]] = 1.0/nG[g]
        
    #Initialize gamma
    gamma = np.ones((len(uniqueGroups),N))
    
    #Create lists to hold temporary variables
    d = []
    Sigma = []
    mu = []
    I = []
    used_ind = []
        
    for k in range(nS):
        d.append(Y[k].shape[0])
        Sigma.append(np.zeros((np.int32(M[k]),np.int32(M[k]),N)))
        mu.append(np.ones((np.int32(M[k]),N)))

    update = np.zeros(nS,dtype=bool) 
    
    for iter in range(p['numIter']):
        print(iter)

        mu_old = copy.deepcopy(mu)

        #Do inference
        for k in range(nS):
            Sigma[k],mu[k],_ = InferenceStep(Y[k],0,A[k],0,gamma,s2[k],0,B_back[k],p)
                   
        for n in range(N):
            gamma[:,n] = 0.0
            for k in range(nS):
                if p['DA']:
                    gamma[:,n] = gamma[:,n] + B_forward[k].dot(Sigma[k][:,n]+np.square(mu[k][:,n]))
                else:
                    gamma[:,n] = gamma[:,n] + B_forward[k].dot(np.diag(Sigma[k][:,:,n])+np.square(mu[k][:,n]))
       
        for k in range(nS):
           update[k] = LA.norm(mu_old[k]-mu[k],ord='fro')/LA.norm(mu_old[k],ord='fro') < 1e-3

        if np.all(update):
            break
        
    return mu

def CreateBatches(Y,numBatches):
    N = Y.shape[1]
    batchSize = np.int(np.floor(np.float(N)/np.float(numBatches)))
    Ybatch = []
    #Initialize batches
    for n in range(numBatches):
        Ybatch.append(Y[:,n*batchSize:(n+1)*batchSize])
    Ybatch[-1] = np.concatenate((Ybatch[-1],Y[:,numBatches*batchSize:N]),axis=1)  
    return Ybatch

def ShuffleData(Y,Ybatch,T,Tbatch,Sigma,mu,gamma,numBatches,batchSize,used_ind,p):
    N = Y[0].shape[1]
    nS = len(Y)
    ind = np.random.permutation(N)
    gamma = gamma[:,ind]
    
    if not p['SS']:
        for ii in range(len(used_ind)):
            used_ind[ii] = np.where(ind == used_ind[ii])[0][0]
    
    for k in range(nS):
        Y[k] = Y[k][:,ind]                        
        for n in range(numBatches):
            Ybatch[k][n] = Y[k][:,n*batchSize:(n+1)*batchSize]
        Ybatch[k][-1] = np.concatenate((Ybatch[k][-1],Y[k][:,numBatches*batchSize:N]),axis=1)
    
    if p['TD']:
        T = T[:,ind]
        for n in range(numBatches):
            Tbatch.append(T[:,n*batchSize:(n+1)*batchSize])
        Tbatch[-1] = np.concatenate((Tbatch[-1],T[:,numBatches*batchSize:N]),axis=1)              
        
    if not p['SufficientStatistics']:
        if p['DA']:
            for k in range(nS):
                mu[k] = np.array(mu[k][:,ind])
                Sigma[k] = np.array(Sigma[k][:,ind])
        else:
            for k in range(nS):
                mu[k] = np.array(mu[k][:,ind])
                Sigma[k] = np.array(Sigma[k][:,:,ind])

    return Y,Ybatch,T,Tbatch,Sigma,mu,gamma,used_ind

def MultimodalDL(Yin,Tin,p):  
    #Yin: list representing input data. Yin[0] is the first dataset, etc. Each column of Yin[0] is a data point
    #Tin: used for task driven learning. Set to 0 for now
    #p: dictionary representing algorithm parameters:
        #p['numOuterIter']: I suggest setting this to 4
        #p['numInnerIter']: I suggest setting this to 250
        #p['batchSize']: batch size for stochastic learning
        #p['n']: list containing desired number of dictionary elements for each modality
        #p['s2_initial']: list containing initial variance values. 
        #p['s2_lowerbound']: list containing final variance values
        #p['s2_decay_factor']: list containing annealing rates for each modality
        #p['TD']: True/False for whether or not to do task-driven learning (use False for now)
        #p['SufficientStatistics']: True/False for whether to use sufficient statistics for entire dataset to update dictionary at each iteration.
        #setting to False will lead to smaller memory footprint
        #p['DA']: True/False for whether to use diagonal approximation when computing covariance matrix. Setting to True leads to significant computational savings.
        #p['SS']: Whether to use subspace learning. Set to False
    np.random.seed()

    #Read in input data and perform initializations
    nS = len(Yin)
    batchSize = np.int(p['batchSize'])
    M = np.zeros(nS,dtype=int)
    s2 = np.array(np.zeros(nS))
    b2 = np.array(np.zeros(nS))
    s2_lowerbound = np.array(np.zeros(nS))
    b2_lowerbound = np.array(np.zeros(nS))
    s2_decay_factor = np.array(np.zeros(nS))
    b2_decay_factor = np.array(np.zeros(nS))
    N = np.int(Yin[0].shape[1])

    Y = []
    #Permute input data to avoid non-randomness
    ind = np.random.permutation(N)
    for k in range(nS):
        M[k] = np.int(p['n'][k])
        s2[k] = np.float(p['s2_initial'][k])
        s2_lowerbound[k] = np.float(p['s2_lowerbound'][k])
        s2_decay_factor[k] = np.float(p['s2_decay_factor'][k])
        Y.append(np.array(Yin[k][:,ind]))

    if p['TD']:
        T = np.array(Tin[:,ind])
        numClasses = T.shape[0]
        for k in range(nS):
            b2[k] = np.float(p['b2_initial'][k])
            b2_decay_factor[k] = np.float(p['b2_decay_factor'][k])
            b2_lowerbound[k] = np.float(p['b2_lowerbound'][k])
    else:
        T = 0
        
    #Initialize group information, including back and forward group operators
    if 'groups' not in p:
        groups = []
        for k in range(nS):
            groups.append(np.array(range(M[k]))+1)
    else:
        groups = copy.deepcopy(p['groups'])

    uniqueGroups = np.array(np.squeeze(np.unique(groups[0])))

    B_back = []
    B_forward = []
    nG = np.zeros(len(uniqueGroups))
    for g in range(len(uniqueGroups)):
        for k in range(nS):
            nG[g] = nG[g]+np.float(np.sum(np.equal(groups[k],uniqueGroups[g])))

    for k in range(nS):
        B_back.append(np.zeros((np.int32(M[k]),len(uniqueGroups))))
        B_forward.append(np.zeros((len(uniqueGroups),np.int32(M[k]))))

        for g in range(len(uniqueGroups)):
            ind = np.where(np.equal(groups[k],uniqueGroups[g]))[0]
            for ii in range(len(ind)):
                B_back[k][ind[ii],g] = np.float(1)
                B_forward[k][g,ind[ii]] = 1.0/nG[g]
        
    #Initialize gamma
    if p['tree']:
        gamma = np.ones((2*np.amax(p['n']),N))
        S = list()
        S.append(np.zeros((gamma.shape[0],p['n'][0])))
        S.append(np.concatenate((np.eye(p['n'][1]),np.eye(p['n'][1]))))

        for m in range(p['n'][1]):
            S[0][m,p['groups'][1][m]] = 1.0
    else:
        gamma = np.ones((len(uniqueGroups),N))
    
    #Determine number of batches
    numBatches = np.int(np.floor(N/np.float(batchSize)))
    
    #Create lists to hold temporary variables
    d = []
    A = []
    Sigma = []
    mu = []
    used_ind = []
    D = []
    W = []
    Yb = []
    Ybatch = []
    Tb = []
    Tbatch = []
    DQ = np.zeros((p['numIter'],nS))
    dLds_hist = np.zeros((p['numIter'],nS))
        
    #Randomly select initial dictionary
    ind_init = np.random.permutation(N)[0:np.int32(np.amax(M))]
    for k in range(nS):
        d.append(Y[k].shape[0])
        A.append(np.array(NormalizeMatrix(Y[k][:,ind_init[0:M[k]]])))

        if p['tree']:
            mu.append(LA.pinv(A[k].dot(S[k].T)).dot(Y[k]))
        else:            
            mu.append(LA.pinv(A[k]).dot(Y[k]))

        if not p['SufficientStatistics']:
            if p['DA']:
                Sigma.append(np.zeros((np.int32(M[k]),N)))
            else:
                if p['tree']:
                    Sigma.append(np.zeros((gamma.shape[0],gamma.shape[0],N)))
                else:
                    Sigma.append(np.zeros((np.int32(M[k]),M[k],N)))
        else:
            Sigma.append(0)
        
        if(p['TD']):
            W.append(NormalizeMatrixRows(np.random.normal(size=(numClasses,M[k]))))
        else:
            W.append(0)

    #Initialize batches
    for k in range(nS):
        Ybatch.append(list())
        for n in range(numBatches):
            Ybatch[k].append(Y[k][:,n*batchSize:(n+1)*batchSize])
            pass
        Ybatch[k][-1] = np.concatenate((Ybatch[k][-1],Y[k][:,numBatches*batchSize:N]),axis=1)  
        
    #Initialize batch
    for k in range(nS):
        Yb.append(Ybatch[k][0])
        
    if p['TD']:
        for n in range(numBatches):
            Tbatch.append(T[:,n*batchSize:(n+1)*batchSize])
            pass
        Tbatch[-1] = np.concatenate((Tbatch[-1],T[:,numBatches*batchSize:N]),axis=1)
        
    stop_updating = np.zeros(nS+nS*p['TD'],dtype=bool) 

    dLds = []
    for k in range(nS):
        if p['TD']:
            dLds.append(np.zeros((N,2)))
        else:
            dLds.append(np.zeros((N,1)))
     
    if p['display']:
        plt.ion()
        fig, axes = plt.subplots(4, nS)
        
    for iter in range(np.int(p['numIter'])):
        print(iter)

        #Save previous estimate of dictionary
        A_old = copy.deepcopy(A)
        if p['TD']:
            W_old = copy.deepcopy(W)

        #Remove redundancies in the dictionary
        if (iter%300 == 0):# and (iter > 0):
            if p['tree']:
                A,gamma,used_ind = RemoveRedundancy_Tree(Y,A,mu,gamma,used_ind,S)
            elif not p['SS']:
                if p['SufficientStatistics']:
                    A,used_ind,gamma=RemoveRedundancy(Yb,A,mu,used_ind,gamma=gamma)
                else:
                    A,used_ind,gamma=RemoveRedundancy(Y,A,mu,used_ind,gamma=gamma)
            else:
                for k in range(nS):
                    if np.sum(np.equal(p['groups'][k],0)) > 1:
                        A[k],gamma,used_ind = RemoveRedundancySBL_Block(Y[k],A[k],mu[k],p['groups'][k],gamma,used_ind)
                A,gamma = RemoveRedundancySS(Y,A,mu,gamma,p)

        #Extract current batch
        for k in range(nS):
            Yb[k] = Ybatch[k][np.mod(iter,numBatches)]
            
        if p['TD']:
            Tb = Tbatch[np.mod(iter,numBatches)]
        
        #Determine number of data points in batch
        Nbatch = Yb[0].shape[1]
        Ninit = np.mod(iter,numBatches)*batchSize
        gamma_batch = np.array(gamma[:,Ninit:Ninit+Nbatch])
        batch_ind = range(Ninit,Ninit+Nbatch)
        
        #Do inference
        for k in range(nS):
            if p['SufficientStatistics']:
                Sigma[k],mu[k],dLds[k][Ninit:Ninit+Nbatch,:] = InferenceStep(Yb[k],Tb,A[k],W[k],gamma_batch,s2[k],b2[k],B_back[k],p)
            else:
                if p['DA']:
                    Sigma[k][:,Ninit:Ninit+Nbatch],mu[k][:,Ninit:Ninit+Nbatch],dLds[k][Ninit:Ninit+Nbatch,:] = InferenceStep(Yb[k],Tb,A[k],W[k],gamma_batch,s2[k],b2[k],B_back[k],p)
                else:
                    if p['tree']:
                        Sigma[k][:,:,Ninit:Ninit+Nbatch],mu[k][:,Ninit:Ninit+Nbatch],dLds[k][Ninit:Ninit+Nbatch,:] = InferenceStepTree(Yb[k],Tb,A[k],W[k],gamma_batch,s2[k],b2[k],S[k],p)
                    else:
                        Sigma[k][:,:,Ninit:Ninit+Nbatch],mu[k][:,Ninit:Ninit+Nbatch],dLds[k][Ninit:Ninit+Nbatch,:] = InferenceStep(Yb[k],Tb,A[k],W[k],gamma_batch,s2[k],b2[k],B_back[k],p)

        if p['SufficientStatistics']:
            offset = 0
        else:
            offset = Ninit

        if p['tree']:
            second_order_stats = [np.zeros((mu[0].shape[0],Nbatch)) , np.zeros((mu[1].shape[0],Nbatch))]
            for n in range(Nbatch):
                for k in range(nS):
                    second_order_stats[k][:,n] = np.diag(Sigma[k][:,:,n+offset])+np.square(mu[k][:,n+offset])

                gamma[:p['n'][1],batch_ind] = 0.5*(second_order_stats[0][0:p['n'][1],:]+second_order_stats[1][:p['n'][1],:])
                gamma[p['n'][1]:,batch_ind] = second_order_stats[1][p['n'][1]:,:]
        else:
            for n in range(Nbatch):
                gamma[:,n+Ninit] = 0.0
                for k in range(nS):
                    if p['DA']:
                        gamma[:,n+Ninit] = gamma[:,n+Ninit] + B_forward[k].dot(Sigma[k][:,n+offset]+np.square(mu[k][:,n+offset]))
                    else:
                        gamma[:,n+Ninit] = gamma[:,n+Ninit] + B_forward[k].dot(np.add(np.diag(Sigma[k][:,:,n+offset]),np.square(mu[k][:,n+offset])))
        
        for k in range(nS):
            if stop_updating[k]:
                continue

            if p['DA']:
                C = np.diag(np.sum(Sigma[k],axis=1))+mu[k].dot(mu[k].T)
            else:
                C = np.sum(Sigma[k],axis=2)+mu[k].dot(mu[k].T)

            if p['SufficientStatistics']:
                YXt = Yb[k].dot(mu[k].T)
            else:
                if p['tree']:
                    YXtS = Y[k].dot(mu[k].T).dot(S[k])
                else:
                    YXt = Y[k].dot(mu[k].T)

            if p['tree']:
                A[k] = NormalizeMatrix(scipy.linalg.lstsq(S[k].T.dot(C).dot(S[k]),YXtS.T)[0].T)
            else:
                A[k] = NormalizeMatrix(scipy.linalg.lstsq(C,YXt.T)[0].T) 
        
        for k in range(nS):
            if p['DA']:
                C = np.diag(np.sum(Sigma[k],axis=1))+mu[k].dot(mu[k].T)
            else:
                C = np.sum(Sigma[k],axis=2)+mu[k].dot(mu[k].T)

            if(p['TD']):
                W[k] = (LA.lstsq(C,np.dot(mu[k],T.transpose()))[0]).transpose()

        for k in range(nS):
            if (LA.norm(A_old[k]-A[k],ord='fro')/LA.norm(A_old[k],ord='fro') < 1e-2) and (not stop_updating[k]):
                if np.sum(dLds[k][:,0]) < 0:
                    s2_lowerbound[k] = s2[k]
                    stop_updating[k] = True
                else:
                    s2[k] = np.maximum(s2[k]*s2_decay_factor[k],s2_lowerbound[k])
                    if s2[k] == s2_lowerbound[k]:
                        stop_updating[k] = True
            if p['TD']:
                if (LA.norm(W_old[k]-W[k],ord='fro')/LA.norm(W_old[k],ord='fro') < 1e-2) and (not stop_updating[k+nS]):
                    if np.sum(dLds[k][:,1]) < 0:
                        b2_lowerbound[k] = b2[k]
                        stop_updating[k+nS] = True
                    else:
                        b2[k] = np.maximum(b2[k]*b2_decay_factor[k],b2_lowerbound[k])
                        if b2[k] == b2_lowerbound[k]:
                            stop_updating[k+nS] = True
                    
        if p['display']:       
            for k in range(nS):
                if k == 0:
                    tmp = DictionaryQuality(p['A'][k],A[k])
                    DQ[iter,k] = np.mean(np.greater(tmp,0.99))
                else:
                    tmpA,tmpgroups = PruneSBL_Dictionary(A[k],mu[k],p['groups'][k],len(p['groups_true'][k]),S[k])
                    if p['tree']:
                        tmp = DictionaryQuality(p['A'][k],tmpA)
                    else:
                        tmp = DictionaryQualitySubspace(p['A'][k],tmpA,p['groups_true'][k],tmpgroups)                
                    DQ[iter,k] = np.mean(tmp)

                for ii in range(4):
                    axes[ii,k].cla()

                axes[0, k].plot(DQ[:iter,k])
                axes[0,k].set_title('s2 = ' + str(s2[k]))
                dLds_hist[iter,k] = np.sum(dLds[k][:,0])
                axes[1,k].plot(dLds_hist[:iter,k])
                axes[1,k].set_title('History of derivative')
                axes[2,k].plot(tmp)
                axes[2,k].set_title('Dictionary Quality')

                if k == nS-1:
                    ip = np.zeros(50)
                    for g in range(50):
                        #print p['A'][0][:,g]
                        m = np.argmax(np.absolute(p['A'][0][:,g].T.dot(A[0])))
                        ind = np.where(np.equal(p['groups'][k],p['groups'][0][m]))[0];
                        if len(ind) > 1:
                            ip[g] = np.absolute(A[k][:,ind[0]].T.dot(A[k][:,ind[1]]));
                    axes[3,k].plot(ip)
                    axes[3,k].set_title('Subspace coherence');  
            fig.canvas.draw()
            time.sleep(0.001)
        
        if np.all(stop_updating):
            break
            
        #Shuffle data
        if np.mod(iter+1,numBatches) == 0:
            if p['TD']:
                Y,Ybatch,T,Tbatch,Sigma,mu,gamma,used_ind = ShuffleData(Y,Ybatch,T,Tbatch,Sigma,mu,gamma,numBatches,batchSize,used_ind,p)
            else:
                Y,Ybatch,_,_,Sigma,mu,gamma,used_ind = ShuffleData(Y,Ybatch,0,0,Sigma,mu,gamma,numBatches,batchSize,used_ind,p)

    return A,W,mu,s2,iter

def PruneSBL_Dictionary(A,mu,groups,M,S):
    G = len(np.unique(groups))
    num_to_remove = len(groups)-M
    tmp = np.mean(np.square(S.T.dot(mu)),axis=1)
    active_ind = np.ones(len(groups))
    active_groups = np.ones(G)
    sorting_ind = np.argsort(tmp)

    #first prune heavily aligned subspaces
    ip = np.zeros(G)
    for g in range(G):
        ind = np.where(np.equal(groups,g))[0];
        if len(ind) > 1:
            ip[g] = np.absolute(np.inner(A[:,ind[0]],A[:,ind[1]]))

    sorted_g = np.argsort(ip)
    sorted_g = sorted_g[::-1]
    
    for g in range(G):
        if ip[sorted_g[g]] > 0.8:
            if active_groups[sorted_g[g]] == 1:
                #Find minimum activated column in the group which will be pruned
                z = np.argmin(tmp[np.where(np.equal(groups,sorted_g[g]))[0]]);
                group_cols = np.where(np.equal(groups,sorted_g[g]))[0];
                ind_prune = group_cols[z];
                active_ind[ind_prune] = 0;
                active_groups[sorted_g[g]] = 0;
                num_to_remove = num_to_remove-1;

                if num_to_remove < 1:
                    break

    while num_to_remove > 0:
        #find minimum activated atom from remaining active groups
        x = 0
        while True:
            ind = sorting_ind[x]
            #Check if this group has been pruned already
            if (active_groups[groups[ind]] == 0) or (np.sum(np.equal(groups,groups[ind])) == 1):
                x = x+1
                continue

            #Prune column
            active_groups[groups[ind]] = 0
            active_ind[ind] = 0
            num_to_remove = num_to_remove-1
            break

    A = A[:,np.where(np.equal(active_ind,1))[0]]
    groups = [groups[ii] for ii in np.where(np.equal(active_ind,1))[0]]
    return A,groups

def ComputeLikelihood(Y,Sigmay):
    N = Y.shape[1]
    L = np.zeros(N)
    for n in range(N):
        L[n] = np.log(LA.det(Sigmay[:,:,n]))+Y[:,n].T.dot(LA.solve(Sigmay[:,:,n],Y[:,n]))

    return np.sum(L)

def MultimodalClassificationAccuracy(T,W,X):
    nS = len(W)
    numClasses = T.shape[0]
    N = T.shape[1]
    success = np.zeros(N)
    I = np.eye(numClasses)
    loss = np.zeros((nS,numClasses))
    
    for n in range(N):
        ind = np.where(T[:,n])[0]
        for k in range(nS):
            loss[k,:] = np.sum(np.square(I-np.matlib.repmat(np.dot(W[k],X[k][:,n])[:,np.newaxis],1,numClasses)),axis=0)
        ind_hat = np.argmin(np.sum(loss,axis=0))
        
        if ind == ind_hat:
            success[n] = 1
        
    return success

def ClassificationAccuracy(T,W,X):
    numClasses = T.shape[0]
    N = T.shape[1]
    success = np.zeros(N)
    I = np.eye(numClasses)
    
    for n in range(N):
        ind = np.where(T[:,n])[0]
        ind_hat = np.argmax(W.dot(X[:,n]))
        
        if ind == ind_hat:
            success[n] = 1
        
    return success

def ComputeLinearClassifier(T,X,nu):
    return scipy.linalg.lstsq(np.concatenate((T,np.zeros((T.shape[0],X.shape[0]),dtype=np.float))),np.concatentate((X,np.sqrt(nu)*np.eye(W.shape[1]))).T)[0].T

def RemoveRedundancy(Y,A,mu,used_ind,**kwargs):
    nS = len(A)
    M = A[0].shape[1]
    
    for k in range(nS):
        for m in range(M):
            ip = np.absolute(np.dot(A[k][:,m].transpose(),A[k]))
            ip[m] = 0
            if((np.amax(ip) > 0.99) | (np.mean(np.absolute(mu[k][m,:])) < 0.03)):
                Error = Y[k]-(np.dot(A[k],mu[k])-np.outer(A[k][:,m],mu[k][m,:]))
                Energy = np.sum(np.square(Error),axis=0)
                ind = np.argsort(Energy)[::-1]

                for n in ind:
                    if n not in used_ind:
                        A[k][:,m] = np.squeeze(NormalizeMatrix(Y[k][:,n]))
                        if 'gamma' in kwargs:
                            kwargs['gamma'][m,:] = 1.0
                        used_ind.append(n)
                        break
    if 'gamma' in kwargs:
        return A,used_ind,kwargs['gamma']
    else:
        return A,used_ind

def RemoveRedundancy_Tree(Y,A,mu,gamma,used_ind,S):
    nS = len(A)
    
    for k in range(nS):
        M = A[k].shape[1]
        StU = S[k].T.dot(mu[k])
        for m in range(M):
            ip = np.absolute(np.dot(A[k][:,m].transpose(),A[k]))
            ip[m] = 0
            if (np.amax(ip) > 0.99) or (np.mean(np.absolute(StU[m,:])) < 0.03):
                Error = Y[k]-A[k].dot(StU)+np.outer(A[k][:,m],StU[m,:])
                Energy = np.sum(np.square(Error),axis=0)
                ind = np.argsort(Energy)[::-1]
                
                for n in ind:
                    if n not in used_ind:
                        A[k][:,m] = np.squeeze(NormalizeMatrix(Y[k][:,n]))
                        gamma[np.where(S[k][:,m])[0],:] = 1.0
                        used_ind.append(n)
                        break

    return A,gamma,used_ind

def RemoveRedundancySS(Y,A,mu,gamma,p):
    nS = len(A)
    unique_groups = np.unique(p['groups'][0])
    G = len(unique_groups)
    for g in range(G):
        for k in range(nS):
            #Check for subspace alignment
            currentSubspace = A[k][:,np.equal(p['groups'][k],g)]
            tmp = np.zeros(G)
            for g_prime in range(G):
                if g_prime == g:
                    continue
                else:
                    otherSubspace = A[k][:,np.equal(p['groups'][k],g_prime)]
                    tmp[g_prime] = SubspaceAngle(currentSubspace,otherSubspace)

            mu_g = mu[k][np.equal(p['groups'][k],g),:]

            if (np.sum(np.absolute(mu_g))/np.float(mu_g.size) < 0.03) or (np.amax(tmp) > 0.98):
                for kk in range(nS):
                    ind = np.setdiff1d(range(p['n'][kk]),np.where(np.equal(p['groups'][kk],g))[0])
                    Error = Y[kk]-np.dot(A[kk][:,ind],mu[kk][ind,:])
                    Energy = np.sum(np.square(Error),axis=0)
                    ind = np.argsort(Energy)[::-1]
                    A[kk][:,np.equal(p['groups'][kk],g)] = NormalizeMatrix(
                        Y[kk][:,ind[0:np.sum(np.equal(p['groups'][kk],g))]])
                gamma[g,:] = 1.0
    return A,gamma

def SubspaceAngle(U,V):
    U = LA.svd(U,full_matrices=False)[0]
    V = LA.svd(V,full_matrices=False)[0]

    M = U.T.dot(V)
    return np.sqrt(np.absolute(LA.det(M.dot(M.T))))

def DictionaryQuality(D,Dhat):
    success = np.zeros(D.shape[1])
    nD = NormalizeMatrix(D)
    nDhat = NormalizeMatrix(Dhat)
    
    for ii in range(D.shape[1]):
        success[ii] = np.amax(np.absolute(nD[:,ii].transpose().dot(nDhat)))
        
    return success

def DictionaryQualitySubspace(D,Dhat,groups_true,groups):
    unique_groups = np.unique(groups_true)
    G = len(unique_groups)
    tmp = np.zeros(G)
    success = np.zeros(G)
    for g in range(G):
        trueSubspace = D[:,np.where(np.equal(groups_true,unique_groups[g]))[0]]
        for g_prime in range(G):
            approximatedSubspace = Dhat[:,np.where(np.equal(groups,unique_groups[g_prime]))[0]]
            tmp[g_prime] = np.real(SubspaceAngle(trueSubspace,approximatedSubspace))
        
        success[g] = np.amax(tmp)
    return success

def RemoveRedundancySBL_Block(Y,A,mu,groups,gamma,used_ind):
    M = A.shape[1]
    G = np.amax(groups)
    for g in range(G):
        Ag = A[:,np.where(np.equal(groups,g))[0]]
        for m in range(Ag.shape[1]):
            ip = np.absolute(Ag[:,m].T.dot(Ag))
            ip[m] = 0
            if np.amax(ip) > 0.9:
                x = np.where(np.equal(groups,g))[0]
                x = np.int(x[m])
                ind = np.setdiff1d(range(M),[x])
                Error = Y-np.dot(A[:,ind],mu[ind,:])
                Energy = np.sum(np.square(Error),axis=0)
                ind = np.argsort(Energy)[::-1]

                for n in ind:
                    if n not in used_ind:
                        A[:,m] = np.squeeze(NormalizeMatrix(Y[:,n]))
                        gamma[g,:] = 1.0
                        used_ind.append(n)
                        break
    return A,gamma,used_ind

def NormalizeMatrix(X):
    new_matrix = np.array(X)
    if(new_matrix.ndim == 1):
        new_matrix = np.reshape(new_matrix,(-1,1))
    new_matrix = sklearn.preprocessing.normalize(new_matrix,axis=0)   
   
    return new_matrix

def NormalizeMatrixRows(X):
    new_matrix = np.array(X)
    if(new_matrix.ndim == 1):
        new_matrix = np.reshape(new_matrix,(-1,1))
    new_matrix = sklearn.preprocessing.normalize(new_matrix,axis=1)   
    return new_matrix

def JointADMM_SparseCoding(Y,A,p):
    #Read in data and perform initializations
    nS = len(Y)
    N = Y[0].shape[1]
    admmLambda = np.float(p['s2_initial'])
    rho = np.float(p['rho'])
    maxIter = p['admmIter']
    M = A[0].shape[1]
    
    #Create lists to hold temporary variables
    d = []
    X = []
    V = []
    Z = []
    I_small = []
    I_big = []
    At = []
    AAt = []
    
    Xtmp = np.zeros((M,nS))
    Vtmp = np.zeros_like(Xtmp)
    r = np.zeros(nS)
    s = np.zeros(nS)
    eps_dual = np.zeros(nS)
    eps_pri = np.zeros(nS)
    
    #Precompute anything that can be procumputed
    for k in range(nS):
        d.append(Y[k].shape[0])
        X.append(np.random.normal(size=(M,N)))
        V.append(np.zeros_like(X[k]))
        Z.append(np.zeros_like(X[k]))
        I_small.append(np.eye(d[k]))
        I_big.append(np.eye(M))
        At.append(np.array(A[k].transpose()))
        AAt.append(np.array(A[k].dot(A[k].transpose())))
        
    for iter in range(maxIter):
        Zprev = copy.deepcopy(Z)
        #ADMM
        for k in range(nS):
            X[k] = np.array((I_big[k]/rho-(At[k]/rho).dot(
                    LA.lstsq(I_small[k]+AAt[k]/rho,A[k]/rho)[0])).dot(
                At[k].dot(Y[k])+rho*(Z[k]-V[k])))
                
        for n in range(N):
            for k in range(nS):
                Xtmp[:,k] = X[k][:,n]
                Vtmp[:,k] = V[k][:,n]
                
            U = np.array(np.add(Xtmp,Vtmp))
            row_sums = np.sqrt(np.sum(np.square(U),axis=1))
            tmp = 1.0-(admmLambda/rho)/row_sums
            tmp[tmp < 0]= 0
            Znew = U*tmp[:,np.newaxis]            
            
            for k in range(nS):
                Z[k][:,n] = Znew[:,k]
                
        for k in range(nS):
            V[k] = V[k]+X[k]-Z[k]
            r[k] = np.sum(np.sum(np.square(X[k]-Z[k])))
            s[k] = np.sum(np.sum(np.square(rho*(Zprev[k]-Z[k]))))
            eps_pri[k] = np.sqrt(M*N)*1e-4+1e-4*np.maximum(LA.norm(X[k],ord='fro'),LA.norm(Z[k],ord='fro'))
            eps_dual[k] = np.sqrt(d[k]*N)*1e-4+1e-4*LA.norm(V[k],ord='fro')
           
        if (all(np.less_equal(r,eps_pri)) & all(np.less_equal(s,eps_dual))):
            break
            
        if np.sqrt(np.sum(r)) > 10*np.sqrt(np.sum(s)):
            rho = rho*2
        elif np.sqrt(np.sum(s)) > 10*np.sqrt(np.sum(r)):
            rho = rho/2
    return X

def MultimodalL1(Yin,Tin,p,TD): 
    #Initialization
    nS = len(Yin)
    maxIter = np.int(p['numOuterIter'])
    M = np.zeros(nS,dtype=int)
    d = np.zeros(nS,dtype=int)
    batchSize = np.int(p['batchSize'])
    t0 = np.float(maxIter)/10.0
    rho = 0.5

    for k in range(nS):
        M[k] = np.int(p['n'][k])
        d[k] = np.int(Yin[k].shape[0])

    A_past = []
    B_past = []
    N = np.int(Yin[0].shape[1])
    A = []
    Y = []
    Ybatch = []
    Yb = []
    used_ind = []
    W = []

    ind = np.random.permutation(N)
    for k in range(nS):
        Y.append(Yin[k][:,ind])
        Yb.append(0)

    if TD:
        T.append(Tin[:,ind])
    else:
        T = 0

    for k in range(nS):
        A.append(np.array(NormalizeMatrix(Y[k][:,np.random.permutation(N)[0:np.int32(M[k])]])))
        if TD:
            W.append(0.01*np.random.normal(size=(T.shape[0],M[k])))
        B_past.append(np.zeros(A[k].shape))
        A_past.append(np.zeros((M[k],M[k])))

    #Determine number of batches
    numBatches = np.int(np.floor(N/np.float(batchSize)))

    for k in range(nS):
        Ybatch.append(CreateBatches(Y[k],numBatches))

    if TD:
        Tbatch = CreateBatches(T,numBatches)

    converge = np.zeros(nS,dtype=bool)
    for iter in range(maxIter):
        print(iter)

        if iter == 0:
            dicIterations = 10
        else:
            dicIterations = 1

        rho_t = min(rho,rho*t0/np.float(iter+1))

        Aold = copy.deepcopy(A)
        if (iter > 1) & (np.mod(iter,100) == 0):
            RemoveRedundancy(Y,A,X,used_ind)

        for k in range(nS):
            Yb[k] = Ybatch[k][np.mod(iter,numBatches)]

        if TD:
            Tb = Tbatch[np.mod(iter,numBatches)]

        Nbatch = Yb[0].shape[1]

        #Solve for X given A
        X = JointADMM_SparseCoding(Yb,A,p)

        #Solve for A given X
        #if batchSize == N: remember to put this back in
        if batchSize == N+1:
            for k in range(nS):
                A[k] = NormalizeMatrix(LA.lstsq(X[k].T,Y[k].T)[0].transpose())
                converge[k] = np.max(np.abs(A[k]-Aold[k])/np.abs(Aold[k]+1e-8)) < 1e-3
        else:
            for k in range(nS):
                A_past[k] = A_past[k]+X[k].dot(X[k].T)/np.float(batchSize) 
                B_past[k] = B_past[k]+Yb[k].dot(X[k].T)/np.float(batchSize)
            
            for k in range(nS):
                D_temp = np.array(A[k])
                for l in range(dicIterations):
                    for j in range(M[k]):
                        if A_past[k][j,j] > 1e-12:
                            D_temp[:,j] = D_temp[:,j]+(B_past[k][:,j]-D_temp.dot(A_past[k][:,j]))/A_past[k][j,j]
                            tempNorm = LA.norm(D_temp[:,j],ord=2)
                            if tempNorm > 1:
                                D_temp[:,j] = np.array(D_temp[:,j]/tempNorm)
                A[k] = np.array(D_temp)
                converge[k] = np.max(np.abs(A[k]-Aold[k])/np.abs(Aold[k]+1e-8)) < 1e-3

        
        if np.all(converge):
            break

        #Shuffle data
        if np.mod(iter+1,numBatches) == 0:
            ind = np.random.permutation(N)
            for k in range(nS):
                Y[k] = Y[k][:,ind]
                for n in range(numBatches):
                    Ybatch[k][n] = Y[k][:,n*batchSize:(n+1)*batchSize]
                    Ybatch[k][-1] = np.concatenate((Ybatch[k][-1],Y[k][:,(numBatches*batchSize):]),axis=1)

            if TD:
                T = T[:,ind]
                for n in range(numBatches):
                    Tbatch.append(T[:,n*batchSize:(n+1)*batchSize])
                    pass
                Tbatch[-1] = np.concatenate((Tbatch[-1],T[:,((numBatches+1)*batchSize):]),axis=1)              

            for ii in range(len(used_ind)):
                used_ind[ii] = np.where(ind == used_ind[ii])[0][0]

        X = JointADMM_SparseCoding(Y,A,p)
    return A,X

def MultimodalOMP_DL(Yin,p):    
    #Initialization
    np.random.seed()
    nS = len(Yin)
    lam = np.zeros(nS)
    for k in range(nS):
        lam[k] = p['s2_initial'][k]
    maxIter = p['numOuterIter']
    S = np.int(p['s'])
    N = Yin[0].shape[1]
    M = np.zeros(nS)
    
    A = []
    A1 = []
    used_ind = []
    Y = []
    Y1 = []
        
    ind = np.random.permutation(N)
    for k in range(nS):
        Y.append(Yin[k][:,ind])
        Y1.append(np.sqrt(lam[k])*Yin[k][:,ind])
        A.append(np.array(NormalizeMatrix(Y[k][:,np.random.permutation(N)[0:p['n'][k]]])))
        A1.append(A[k])
       
    converge = np.zeros(nS)
    for iter in range(maxIter):
        Aold = copy.deepcopy(A)

        print(iter)
        if (iter > 0) & (np.mod(iter,100) == 0):
            RemoveRedundancy(Y,A,X,used_ind)
        
        for k in range(nS):
            A1[k] = np.array(A[k]*np.sqrt(lam[k]))          
    
        X = MultimodalOMP(Y1,A1,S)
        Xold = copy.deepcopy(X)
        
        for k in range(nS):
            A[k],X[k] = KSVD_DictionaryUpdateStage(Y[k],A[k],X[k],used_ind)
            converge[k] = LA.norm(Aold[k]-A[k],ord='fro')/LA.norm(Aold[k],ord='fro')
        
        if np.all(converge < 1e-3):
            break
        
    return A

def MultimodalOMP(Y,D,S):
    nS = len(Y)
    N = Y[0].shape[1]
    M = D[0].shape[1]
    
    d = []
    X = []
    
    for k in range(nS):
        d.append(Y[k].shape[0])
        X.append(np.zeros((M,N)))
   
    Yn = np.zeros((np.sum(d),N))
    Xn = np.zeros((M*nS,N))

    for n in range(N):
        for jj in range(nS):
            Yn[np.int(np.sum(d[0:jj])):np.int(np.sum(d[0:jj+1])),n] = Y[jj][:,n]

    for n in range(N):
        Xn[:,n] = DCS_SOMP(Yn[:,n],D,S,d)

    for n in range(N):
        for jj in range(nS):
            X[jj][:,n] = Xn[jj*M:(jj+1)*M,n]
            
    return X

def DCS_SOMP(Yn,D,S,d):
    nS = len(d)
    M = D[0].shape[1]
    Lambda = []
    Q = []
    X = []
    I = []
    Y = []
    
    for k in range(nS):
        Y.append(np.array(Yn[np.int(np.sum(d[0:k])):np.int(np.sum(d[0:k+1]))]))
        Q.append(np.zeros((d[k],S)))
        X.append(np.zeros(M))
        I.append(np.eye(d[k]))
        
    R = copy.deepcopy(Y)
        
    for s in range(S):
        ip = np.zeros((S,M))
        for k in range(nS):
            ip[k,:] = np.abs(np.dot(D[k].T,R[k]))
        
        ind = np.argmax(np.sum(ip,axis=0))
        Lambda.append(ind)
    
        for k in range(nS):
            if s == 0:
                Q[k][:,s] = np.squeeze(NormalizeMatrix(D[k][:,Lambda[-1]]))
            elif s== 1:
                Q[k][:,s] = np.squeeze(
                    NormalizeMatrix(
                        (I[k]-np.outer(Q[k][:,0:s],Q[k][:,0:s])).dot(D[k][:,Lambda[-1]])))
            else:
                Q[k][:,s] = np.squeeze(
                    NormalizeMatrix(
                        (I[k]-Q[k][:,0:s].dot(Q[k][:,0:s].T)).dot(D[k][:,Lambda[-1]])))
            R[k] = R[k]-np.outer(Q[k][:,s],Q[k][:,s]).dot(Y[k])
    
    for k in range(nS):
        X.append(np.zeros(M))
        X[k][Lambda] = LA.lstsq(D[k][:,Lambda],Y[k])[0]
    
    Xn = np.zeros(nS*M)
    for jj in range(nS):
        Xn[jj*M:(jj+1)*M] = X[jj]
    
    return Xn

def KSVD_DictionaryUpdateStage(Y,Din,Xin,used_ind):
    D = np.array(Din)
    X = np.array(Xin)
    M = D.shape[1]
    for k in range(M):
        ind = np.nonzero(X[k,:])[0]
        if(len(ind) == 0):
            Error = Y-np.dot(D,X)
            Energy = np.sum(np.square(Error),axis=0)
            indE = np.argsort(Energy)[::-1]
            for n in indE:
                if n not in used_ind:
                    D[:,k] = np.squeeze(NormalizeMatrix(Y[:,n]))
                    used_ind.append(n)
                    break
            break
        
        Ek = Y-D.dot(X)+np.outer(D[:,k],X[k,:])
        Ek = np.array(Ek[:,ind])
        if(len(ind) == 1):
            s = LA.norm(Ek,ord=2)
            D[:,k] = np.squeeze(Ek/s)
            X[k,ind] = s
        else:
            U,s,Vh = LA.svd(Ek,full_matrices=False)
            D[:,k] = U[:,0]
            X[k,ind] = s[0]*Vh[0,:]
    return D,X

def OneHot(x):
    C = np.unique(x).size
    N = x.size
    y = np.zeros((C,N))
    for n in range(N):
        y[x[n],n] = 1

    return y

def PreProcessData(X,Z):
    mu = np.mean(X,axis=1)
    return NormalizeMatrix((X.T-mu).T),NormalizeMatrix((Z.T-mu).T)

def main():
    #argv[1]: input file
    #argv[2] = parameter file
    #argv[3] = TD?
    #argv[4] = Subspace?
    #argv[5] = what kind of leanring (sbldl,...)
    #argv[6] = output file

    #Load the data
    file_contents_data = scipy.io.loadmat(sys.argv[1])
    p = dict()

    if sys.argv[2].lower() == 'msbdl':
        for k in range(3,len(sys.argv),2):
            if sys.argv[k].lower() == 'display':
                if sys.argv[k+1].lower() == 'true':
                    p['display'] = True
                else:
                    p['display'] = False
            elif sys.argv[k].lower() == 'tree':
                if sys.argv[k+1].lower() == 'true':
                    p['tree'] = True
                else:
                    p['tree'] = False
            elif sys.argv[k].lower() == 'file_save':
                f_save = sys.argv[k+1]
            elif sys.argv[k].lower() == 'numiter':
                p['numIter'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'batchsize':
                p['batchSize'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'n1':
                p['n'] = []
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'n2':
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'n3':
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_initial1':
                p['s2_initial'] = []
                p['s2_initial'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_initial2':
                p['s2_initial'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_initial3':
                p['s2_initial'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_lowerbound1':
                p['s2_lowerbound'] = []
                p['s2_lowerbound'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_lowerbound2':
                p['s2_lowerbound'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_lowerbound3':
                p['s2_lowerbound'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_decay_factor1':
                p['s2_decay_factor'] = []
                p['s2_decay_factor'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_decay_factor2':
                p['s2_decay_factor'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_decay_factor3':
                p['s2_decay_factor'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'b2_decay_factor1':
                p['b2_decay_factor'] = []
                p['b2_decay_factor'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'b2_decay_factor2':
                p['b2_decay_factor'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'b2_decay_factor3':
                p['b2_decay_factor'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'td':
                if sys.argv[k+1].lower() == 'false':
                    p['TD'] = False
                else:   
                    p['TD'] = True
            elif sys.argv[k].lower() == 'ss':
                if sys.argv[k+1].lower() == 'false':
                    p['SS'] = False
                else:   
                    p['SS'] = True
            elif sys.argv[k].lower() == 'groups':
                p['groups'] = list()
                if sys.argv[k+1].lower() == 'a':
                    p['groups'].append(range(50))
                    p['groups'].append(range(50)+range(50))
                elif sys.argv[k+1].lower() == 'b':
                    p['groups'].append(range(50))
                    p['groups'].append(range(50)+range(50))
                elif sys.argv[k+1].lower() == 'c':
                    p['groups'].append(range(50))
                    p['groups'].append(range(50)+range(50))
            elif sys.argv[k].lower() == 'sufficientstatistics':
                if sys.argv[k+1].lower() == 'false':
                    p['SufficientStatistics'] = False
                else:
                    p['SufficientStatistics'] = True
            elif sys.argv[k].lower() == 'da':
                if sys.argv[k+1].lower() == 'false':
                    p['DA'] = False
                else:
                    p['DA'] = True
            elif sys.argv[k].lower() == 'normalize_data':
                if sys.argv[k+1].lower() == 'true':
                    for j in range(len(file_contents_data['Y'][0])):
                        file_contents_data['Y'][0][j],_ = PreProcessData(file_contents_data['Y'][0][j],file_contents_data['Y'][0][j])

        A_hat,W_hat,X_hat,s2,total_iter = MultimodalDL(file_contents_data['Y'][0],0,p)
        scipy.io.savemat(f_save,dict(X_hat=X_hat,A_hat=A_hat,W_hat=W_hat,s2=s2,total_iter=total_iter))
    elif sys.argv[2].lower() == 'jl0dl':
        for k in range(3,len(sys.argv),2):
            if sys.argv[k].lower() == 'file_save':
                f_save = sys.argv[k+1]
            elif sys.argv[k].lower() == 'numouteriter':
                p['numOuterIter'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'numinneriter':
                p['numInnerIter'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'batchsize':
                p['batchSize'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'n1':
                p['n'] = []
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'n2':
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'n3':
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_initial1':
                p['s2_initial'] = []
                p['s2_initial'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_initial2':
                p['s2_initial'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_initial3':
                p['s2_initial'].append(np.float(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's':
                p['s'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'normalize_data':
                if sys.argv[k+1].lower() == 'true':
                    for j in range(len(file_contents_data['Y'][0])):
                        file_contents_data['Y'][0][j],_ = PreProcessData(file_contents_data['Y'][0][j],file_contents_data['Y'][0][j])

        A_hat = MultimodalOMP_DL(file_contents_data['Y'][0],p)
        scipy.io.savemat(f_save,dict(A_hat=A_hat))
    elif sys.argv[2].lower() == 'jl1dl':
        for k in range(3,len(sys.argv),2):
            if sys.argv[k].lower() == 'file_save':
                f_save = sys.argv[k+1]
            elif sys.argv[k].lower() == 'numouteriter':
                p['numOuterIter'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'batchsize':
                p['batchSize'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'n1':
                p['n'] = []
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'n2':
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 'n3':
                p['n'].append(np.int(sys.argv[k+1]))
            elif sys.argv[k].lower() == 's2_initial':
                p['s2_initial'] = np.float(sys.argv[k+1])
            elif sys.argv[k].lower() == 'admmiter':
                p['admmIter'] = np.int(sys.argv[k+1])
            elif sys.argv[k].lower() == 'rho':
                p['rho'] = np.float(sys.argv[k+1])
            elif sys.argv[k].lower() == 'normalize_data':
                if sys.argv[k+1].lower() == 'true':
                    for j in range(len(file_contents_data['Y'][0])):
                        file_contents_data['Y'][0][j],_ = PreProcessData(file_contents_data['Y'][0][j],file_contents_data['Y'][0][j])

        A_hat,X_hat = MultimodalL1(file_contents_data['Y'][0],0,p,False)
        scipy.io.savemat(f_save,dict(A_hat=A_hat,X_hat=X_hat))

if __name__ == "__main__":
    main()
