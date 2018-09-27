# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:38:07 2017


main method for the implementation of Markov Cluster Algorithm

Includes toy dataset
and import functionality for Facebook data
both for use with MCL

Calling the algorithm:
mcl.MCL(A,e=2,r=2,maxKeep=200,maxrep=100,minrep=5)

parameters
    A       adjacency matrix
    e       expansion parameter
    r       inflation parameter
    maxKeep pruning parameter
    maxrep  maximum number of iterations
    minrep  minimum number of iterations

@author: Madeleine Duretete
"""

import mcl
import pandas as pd
import numpy as np

### parameters
e = 2
r = 2
maxKeep = 200
maxrep = 100
minrep = 5


### Example using toy matrix
A = mcl.giveMat(6)
print("Adjacency matrix ")
print(A)
mcl.MCL(A,e,r,maxKeep,maxrep,minrep)


### import data

data = pd.read_csv('facebook_combined.txt',delim_whitespace=True)

# and create adjacency matrix
data = np.transpose(np.array(data))
data1 = data[0]
data2 = data[1]

size = len(data1)
range1 = np.amax(data1) + 1
range2 = np.amax(data2) + 1
datarange = max(range1,range2)  

am = np.zeros((datarange,datarange))

for i in range(size):
    x = data1[i]
    y = data2[i]
    am[x][y] = 1
    am[y][x] = 1


### run
#print(am)
#mcl.MCL(am,e,r,maxKeep,maxrep,minrep)
