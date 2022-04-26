from cmath import pi
from logging import NullHandler
from math import inf
from re import VERBOSE
from termios import VERASE
from numpy import conjugate, empty
import numpy as np
import math
import sys
import warnings

def pdist2(X,Y, metric):
    if metric  == None:
        metric = 0

    if metric == 0 or metric == "squclidean":
        D = distEucSq(X,Y)
    # elif metric == "euclidean":
    #     D = np.sqrt(distEucSq(X,Y))
    # elif metric == "L1":
    #     D = distL1(X,Y)
    # elif metric == "cosine":
    #     D = distCosine(X,Y)
    # elif metric == "emd":
    #     D = distEmd(X,Y)
    # elif metric == "chisq":
    #     D = distChiSq(X,Y)
    else:
        sys.exit("Error (pdist2 metric is unreasonable)")

    return D

def distEucSq(X,Y):
    m = len(X)
    n = len(Y)

    X = np.array(X)
    Y = np.array(Y)
    XX = np.sum(X*X, axis=1)
    YY = np.sum(np.conjugate(Y.T)*np.conjugate(Y.T), axis=0)

    XX = XX.reshape(len(XX),1)
    XX_tile = np.tile(XX, n)
    YY_tile = np.stack([YY for _ in range(m)], axis=0)
    # print(np.shape(XX_tile),np.shape(YY_tile))
    # print(np.shape(np.dot(X,np.conjugate(Y.T))))

    D = XX_tile + YY_tile - 2*( np.dot(X, np.conjugate(Y.T)) )
    if float("nan") in D:
        print("There is NaN in function of distEucSq")
    #D = XX[:,np.ones(n)] + YY[np.ones(m),:] - 2*X*np.conjugate(Y.T)
    return D


# def OPW(X,Y, *l1, *l2, *delta, VERBOSE):
def OPW(X,Y, l1, l2, delta, VERBOSE):

    # if kwargs.get("l1",1) is empty:
    #     l1 = 50
    # if kwargs.get("l2",2) is empty:
    #     l2 = 0.1
    # if kwargs.get("delta",3) is empty:
    #     delta = 1
    # if kwargs.get("VERBOSE",4) is empty:
    #     VERBOSE = 0

    tolerance = .5e-2
    maxIter = 20
    p_norm = inf

    N = len(X)
    M = len(Y)
    dim = len(X[0])
    if len(Y[0]) != dim:
        print('The dimensions of instances in the input sequences must be the same!')
    
    P = np.zeros((N,M))
    mid_para = np.sqrt(1/(N**2) + 1/(M**2))
    for i in range(N):
        for j in range(M):
            d = abs( (i+1)/N - (j+1)/M )/mid_para
            # if math.isnan(d):
            #     print("d=NaN")
            # elif d==0:
            #     print("d=0")
            # elif math.isinf(d):
            #     print("d=inf")
            P[i][j] = math.exp(-(d**2)/(2*(delta**2))) / (delta*np.sqrt(2*pi))
            if math.exp(-(d**2)/(2*(delta**2))) / (delta*np.sqrt(2*pi)) == 0:
                print(i,j, " is zero")

    S = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            S[i][j] = l1 / (((i+1)/N - (j+1)/M)**2 + 1)
    
    D = pdist2(X,Y,"squclidean")
    D = D / max(list(map(lambda x: max(x), D)))
    # D = D / (10**2)
    # print("(S - D)/l2=",(S - D)/l2)
    # print("S=",S)
    # print("D=",D)
    K = P * np.exp((S - D)/l2)
    if 0 in P:
        print("P three is 0")
    a = np.ones(N)/N
    a = a.reshape(len(a),1)
    b = np.ones(M)/M
    b = b.reshape(len(b),1)

    ainvK = K / a
    # if 0 in K:
    #     print("K three is 0")
    compt = 0
    u = np.ones(N)/N
    u = u.reshape(len(u),1)
    # if 0 in np.dot(np.conjugate(K.T), u):
    #     print("there is zero")
   
    warnings.simplefilter('ignore', category=RuntimeWarning) 
    # print("K=",K)
    # print("u=",u)
    while compt < maxIter:
        u = 1 / np.dot(ainvK,  b / np.dot(np.conjugate(K.T), u))
        compt += 1
        if compt%20==1 or compt==maxIter:
            # v = b / (np.conjugate(K.T) * u)
            v = b / np.dot(np.conjugate(K.T), u)
            u = 1 / np.dot(ainvK ,v)
            # print(v)
            Criterion = np.linalg.norm(sum(abs(v * (np.conjugate(K.T)-b))), ord=p_norm)
            # print(" Criterion: ",str(Criterion))
            # print(" tolerance: ", str(tolerance))
            if Criterion < tolerance or math.isnan(Criterion):
                break

            compt += 1
            if VERBOSE > 0:
                print(["Iteration :",str(compt)," Criterion: ",str(Criterion)])

    U = K * D
    dis = sum(u * np.dot(U, v))
    T = np.conjugate(v.T) * (u*K)

    return dis, T
