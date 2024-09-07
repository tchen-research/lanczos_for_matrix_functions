#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import mpmath as mp

def get_clusters(Theta,cluster_width):
    """
    finds clusters of Ritz values
    
    Input
    -----
    Theta : Ritz values
    cluster_width : specified width of cluster
    
    Returns
    -------
    clusters : python list of numpy array of clusters
    
    Note
    ----
    we assume Theta is sorted so that min|\theta_i - \theta_j| will be one of the neighbors
    """    
    
    d_Theta = np.ediff1d(Theta,to_begin=0) > cluster_width

    # label clusters
    cluster_labels = np.cumsum(d_Theta)
    
    # identify clusters
    # note that this is inefficient since we search the whol array for each k
    return [np.where(cluster_labels==k)[0] for k in range(max(cluster_labels)+1)]

def mgs(A):
    """
    return orthonormal basis for columns of A using modified gram-schmidt
    """
    m,n = np.shape(A)
    Q = np.copy(A)

    for j in range(n):
        for i in range(j):
            Q[:,j] -= (Q[:,i]@Q[:,j]) * Q[:,i]
        Q[:,j] /= np.linalg.norm(Q[:,j])
        
    return Q

def orth_proj(W,q):
    """
    compute (I-WW*)q
    """
    
    for j in range(W.shape[1]):
        c = W[:,j]@q
        q -= c*W[:,j]
        
    return q

def extend_t(A,alpha,beta,Z,extended_precision = np.float128,verbose=0):
    """
    extends T to a tridiagonal matrix \tilde{T} with eigenvalues in small intervals about those of A
    
    Input
    -----
    A : Original matrix
    Z : lanczos vectors approximately satisfying AZ = ZT and Z_j^TZ_{j+1} = 0
    alpha : diagonal of tridiagonal matrix T
    beta : off diagonal of tridiagonal matrix T
    
    Returns
    -------
    Alpha : diagonal of extended tridiagonal matrix \tilde{T}
    Beta : off diagonal of extended tridiagonal matrix \tilde{T}
    Q : 
    
    """
    
    n = len(A)
    J = len(alpha)
    
    T = sp.sparse.diags([alpha,beta,beta],[0,1,-1],shape=(J+1,J))
    eps_1 = np.max(np.linalg.norm(A@Z[:,:-1]-Z@T,axis=0))
    eps_2 = np.max(np.abs(np.einsum('ji,ji->i',beta*Z[:,:-1],Z[:,1:])))
    
    if verbose>0:
        print(f'three term error: eps_1 = {eps_1}')
        print(f'orthogonality: eps_2 = {eps_2}')

    if verbose>0:
        print(f'\nimporting matrices to extended precision {extended_precision}')
    
    A = A.astype(extended_precision)
    T = T.A.astype(extended_precision)[:-1,:]
    Z = Z.astype(extended_precision)    
    beta_J = beta[-1].astype(extended_precision)    
        
    # note that these computations are done in float64
    eig_A,_ = sp.linalg.eigh(A) 
    Theta,S = sp.linalg.eigh(T)
    
    norm_A = eig_A[-1]
    
    # make bottom entries of S positive
    S *= np.sign(S[-1])
        
    # define ritz vectors
    Y = Z[:,:-1]@S
    
    # find clusters of ritz values
    cluster_width = np.sqrt(max(eps_1,eps_2))*norm_A
    clusters = get_clusters(Theta,cluster_width)
        
    # combine clustered ritz vectors
    Y_C = np.zeros((n,len(clusters)),dtype=extended_precision)
    w_C = np.zeros(len(clusters),dtype=extended_precision)
    for j,cluster in enumerate(clusters):
        w_C[j] = np.linalg.norm(S[J-1,cluster])
        Y_C[:,j] = Y[:,cluster]@S[J-1,cluster] / w_C[j]
    
    #######################################################################
    #
    # TODO:
    #    
    # sort cluters based on projection onto z_{J+1}
    # 
    # its not clear what the best way to choose "unconverged" vectors is
    #
    #######################################################################
    
    proj_C = Y_C.T@Z[:,-1]
    for j in range(len(clusters)):
        proj_C[j] /= np.linalg.norm(Y_C[:,j])
    
    proj_sort = np.argsort(np.abs(proj_C)) 

    unconverged_idx = proj_sort[:0] # can loop here to determine m to balance projections of z_J and z_Jp1
#    unconverged_idx = proj_C *beta_J < cluster_width

    # check which vectors are converged
#    unconverged_idx = w_C * beta_J > cluster_width
        
    # construct \hat{Y}_{m}
    Y_hat = Y_C[:,unconverged_idx]

    m = Y_hat.shape[1]
    
    # allocate space for extended matrices
    Alpha = np.zeros(J+n-m,dtype=extended_precision)
    Beta = np.zeros(J+n-m,dtype=extended_precision)
    Q = np.zeros((n,J+n-m),dtype=extended_precision)

    Alpha[:J] = alpha
    Beta[:J] = beta
    Q[:,:J+1] = Z

    if verbose>0:
        print(f'\nthere are m={m} unconverged ritz values')
        print(f'algorithm will iterate for {n-m} iterations to find extended matrix')

    # Wk = [W,q_{J+1}, ..., q_{J+k}]
    Wk = np.zeros((n,n+1),dtype=extended_precision) 

    # construct W, orthonormal basis for range Y_hat
    Wk[:,:m] = mgs(Y_hat)
    
    if verbose>0:
        print(f'\nprojection of z_{{J}} onto range(Y_hat)^perp: {beta_J*np.linalg.norm(Z[:,J-1]-Wk[:,:m]@Wk[:,:m].T@Q[:,J-1]):1.3e}')
        print(f'projection of z_{{J+1}} onto range(Y_hat): {beta_J*np.linalg.norm(Wk[:,:m].T@Q[:,J]):1.3e}')

    Wk[:,m] = Q[:,J] - Wk[:,:m]@(Wk[:,:m].T@Q[:,J])
    Wk[:,m] /= np.linalg.norm(Wk[:,m])

    if verbose>0:
        print(f'orthogonality of W_{1}: {np.linalg.norm(Wk[:,:m+1].T@Wk[:,:m+1]-np.eye(m+1)):1.3e}')

    for k in range(n-m):
        # update Q
        Q[:,J+k] = Wk[:,m+k]

        # define new vector to satisfy three term recurrence
        Alpha[J+k] = (A@Q[:,J+k] - Beta[J+k-1] * Q[:,J+k-1]) @ Q[:,J+k]
        q_tt = A@Q[:,J+k] - Alpha[J+k]*Q[:,J+k] - Beta[J+k-1]*Q[:,J+k-1]         

        # orthogonalize q_tt against Wk (projected component should be small)
        Wk[:,m+k+1] = q_tt - Wk[:,:m+k+1]@(Wk[:,:m+k+1].T@q_tt)
        Beta[J+k] = np.linalg.norm(Wk[:,m+k+1])
        Wk[:,m+k+1] /= Beta[J+k]

        if verbose>0:
            print(f'\nstep {J+k}')
            print(f'projection onto W_{k+2}:{np.linalg.norm(Wk@(Wk.T@q_tt)):1.3e}') 
            print(f'orthogonality of W_{k+2}: {np.linalg.norm(Wk[:,:m+k+2].T@Wk[:,:m+k+2]-np.eye(m+k+2)):1.3e}')

    if verbose>0:
        print(f'norm of final extended vector: {Beta[-1]:1.3e}')

    return Alpha,Beta,Q