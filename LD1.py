import abc
import numpy as np
from scipy.special import gammaln, psi
from scipy.misc import logsumexp
from scipy.spatial.distance import cosine 
import scipy as sp
from matplotlib import pyplot as plt 
import statsmodels.api as sm




def get_W_1(data,option="euclid",normalise=True):
    assert option in ["euclid","cosine"]

    dim = len(data)
    W_1 = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1,dim):
            if option == "euclid":
                W_1[i,j] = np.exp(- np.linalg.norm(data[i] - data[j],ord=2))
            else:
                W_1[i,j] = cosine(data[i],data[j])
    # Since the distances are symetric
    W_1 += W_1.T
    # Normalize the matrix
    if normalise:
        row_sums = W.sum(axis=1)
        W_1 = W_1 / row_sums[:, np.newaxis]
        
    return W_1

def get_W_real(W,threshold=15):
    #get distribution from W 
    K = 30
    N = W_1.shape[0]
    # Remove zeros on diagonals from matrix
    W_no_diag = np.setdiff1d(np.concatenate(W),[0.])
    # Univariate KDE with Gaussian
    kde = sm.nonparametric.KDEUnivariate(W_no_diag)
    #kde.fit() 

    # Printing the densitry fit to histogram
    #print("Value threshold: %s" % np.percentile(kde.support,q=threshold))
    #fig, ax = plt.subplots(figsize=(8, 6))
    #ax.hist(W_no_diag, bins=K, density=True, lw=0, alpha=0.5)
    #ax.plot(kde.support, kde.density, lw=3, label='Kernel Density Estimation', zorder=10)
    #plt.show()

    # make binary matrix A from W
    A_1= W_1 < np.percentile(kde.support,q=threshold)
    A_1 = A_1.astype(int)
    W = A*W_1
    
    return W

def A_gph(self, W)
     """
     compute the adjacency matrix for the random graph using the Bernulli distribution A_{i,j} ~ Bern(W_{i,j})
     """
     A = self.A
     W = self.A
     n = self.W[n1,n2]
     p = self.p    #kaj dat za p??

     for n1 in range(self.N):
            for n2 in range(self.N):
                A[n1,n2] = np.random.binomial(n,p,size=none)

    return A


