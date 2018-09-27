
"""
Created on Sat Jul  8 00:07:55 2017

Contains 

@author: mdure
"""

import numpy as np
import random
import pandas as pd


# Create toy adjacency (transition) matrix
def giveMat(n):
    mat = np.zeros((n,n))
    for i in range(1,n):
        newrow = random.sample(range(i), random.choice(range(i)))
        mat[i][newrow] = 1
        for j in newrow:
            mat[j][i] = 1
         
    return mat

# Add self loops to
# avoid simple paths of odd (even) length taking higher weight
# on odd (even) powers of expansion
def self_loop(A):
    for i in range(len(A)): A[i][i] = 1


# Normalize

def normalize(A):
    new = A
    colsums = [0]*len(new)
    for i in range(len(new)):
        colsums = new[i] + colsums
    zeros = colsums == 0
    colsums[zeros] = 1
    for i in range(len(new)):
        new[i] = np.divide(new[i],colsums)
        new = np.around(new,4)
    return new

# Expand by taking e-th power of mat
# method: http://people.revoledu.com/kardi/tutorial/LinearAlgebra/MatrixPower.html
def expand(A,e):
    w, P = np.linalg.eig(A)
    try:
        w = np.around(w,4)
        De = np.diag(w**e)
        pre = np.dot(P,np.diag(w))
        Pinv = np.linalg.solve(pre,A)
        first = np.dot(De,Pinv)
        result = np.dot(P,first)
        #print(np.around(result,4))
        return np.around(result,4)
    except np.linalg.LinAlgError:
        return np.around(np.linalg.matrix_power(A,e),4)

# Inflate by taking inflation with r 
def inflate(A,r,maxKeep):
    prune(A,maxKeep)
    return normalize(np.power(A,r))

# Set small values to zero
def prune(mat, keep):
    if len(mat) >= keep:
        for i in range(len(mat)):
            col = mat[:,i]
            sortedCol = np.sort(col,kind='heapsort')
            cutoff = sortedCol[keep-1]
            for j in range(len(mat)):
                if mat[:,i][j] <= cutoff:
                   mat[:,i][j] = 0
    return   


# Repeat till convergence
def iterate(A,e,r,maxKeep,maxrep,minrep):
    previous = A
    current = A
    for i in range(maxrep):
        if not stop(current,previous,i,maxrep,minrep):
            previous = current
            #print("rep:" + str(i+1))
            #print("Expansion")
            M = expand(current,e)
            #print(M)
            M = inflate(M,r,maxKeep)
            #print("Inflation")
            #print(M)
            current = M
        else: break
    return current

# Detect convergence 
def stop(current,prev,it,maxrep,minrep):
    #print(str(it))
    if it>minrep:
        diff = np.sum(np.absolute(np.subtract(current,prev)))
        if diff==0 or it>maxrep:
            print("ended at difference " + str(diff) + " after " + str(it) + " iterations")
            return True
        else: return False


# Interpret, discover clusters
def clusters(M):
    # attractors
        # Have at least one pos flow val
        # within corresp row in conv mat
        # Each attract verts with pos values in row
    # attracted vertices

    
    # check for case when a node in mult clusters,
    # vert pulled exactly equally,
    # case when clusters isomorphic
    
    clust = []
    for i in range(len(M)):
        cl = list(np.where(M[i]>0)[0])
        if len(cl)>0:
            clust.append(cl)
    clust = sorted(clust)
    clust = [clust[i] for i in range(len(clust)) if i == 0 or clust[i] != clust[i-1]]
    print('Clusters:')
    print(clust)
    return clust


# Full call to algorithm on matrix A
def MCL(A,e=2,r=2,maxKeep=200,maxrep=100,minrep=5):
    #maxkeep should be 500-1500
    self_loop(A)
    A = normalize(A)
    end = iterate(A,e,r,maxKeep,maxrep,minrep)
    clusters(end)
    return







