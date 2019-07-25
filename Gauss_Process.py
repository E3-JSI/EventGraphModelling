import pdb 
import numpy as np 
from scipy.spatial.distance import cosine 
import networkx as nx 
from matplotlib import pyplot as plt 
import statsmodels.api as sm
import scipy.stats as ss 

def get_W(data,option="euclid",normalise=True):
    assert option in ["euclid","cosine"]

    dim = len(data)
    W = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1,dim):
            if option == "euclid":
                W[i,j] = np.linalg.norm(data[i] - data[j],ord=2)
            else:
                W[i,j] = cosine(data[i],data[j])
    # Since the distances are symetric
    W += W.T
    # Normalize the matrix
    if normalise:
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]
        
    return W

def get_A_real(W,threshold=15):
    # get distribution from W 
    K = 30
    N = W.shape[0]
    # Remove diagonals from matrix (self connections in the graph)
    W_no_diag = np.delete(np.concatenate(W),[i*N+i for i in range(N)])
    # Univariate KDE with Gaussian
    kde = sm.nonparametric.KDEUnivariate(W_no_diag)
    kde.fit() 
    #kde2 = ss.gaussian_kde(W_no_diag)

    # Printing the densitry fit to histogram
    limit = np.percentile(kde.evaluate(np.linspace(np.min(W_no_diag),np.max(W))),threshold)

    print("Value threshold: %s" % limit)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(W_no_diag, bins=N, density=True,lw=0,alpha=0.5)
    ax.plot(kde.support, kde.density, lw=3, label='Kernel Density Estimation')
    #ax.plot(kde.support,kde2.evaluate(kde.support),label="KDE2")
    ax.plot()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(kde.support, kde.density/np.max(kde.density), lw=3, label='Kernel Density Estimation')
    #ax.plot(kde.support,kde2.evaluate(kde.support),label="KDE2")
    ax.plot()
    plt.show()

    # make binary matrix A from W
    A = W < limit
    A = A.astype(int)
    
    # plot the new Graph
    G = nx.from_numpy_matrix(A)
    nx.draw(G,with_labels=True,with_edges=True)
    plt.show()

    return A 

def get_synthetic_W(option="simple",dim=10):
    assert option in ["simple","gauss","mixed gauss"], "Please choose option from: 'simple', 'gauss' or 'mixed gauss'"
    # Set the random seed
    seed = 1122334455
    np.random.seed(seed)
    # Sample according to options
    if option == "simple": 
        tmp = np.random.random_integers(low=0,high=1,size=(dim,dim))
    elif option == "gauss":
        # To make sure the values are not negative set mean to 2 and var to 1
        tmp = np.random.normal(2,1,(dim,dim))
    elif option == "mixed gauss": 
        x_grid = np.linspace(-1,3,dim*dim)
        tmp = 0.8 * ss.norm(1, 1).pdf(x_grid) + 0.2 * ss.norm(2, 0.3).pdf(x_grid)
        tmp = tmp.reshape(dim,-1)
    # Make the matrix symetric     
    W = np.tril(tmp) + np.tril(tmp,-1).T 
   
    return W

class LatentDistanceAdjacencyDistribution():
    """
    l_n ~ N(0, sigma^2 I)
    A_{n', n} ~ Bern(\sigma(-||l_{n'} - l_{n}||_2^2))
    """
    def __init__(self, N, L= None, 
                 dim=2, sigma=1.0, mu0=1.0, mu_self=0.0):
        self.N = N
        self.dim = dim
        self.sigma = sigma
        self.mu_0 = mu0
        self.mu_self = mu_self
        if L is not None:
            self.L = L
        else: 
            self.L = np.sqrt(self.sigma) * np.random.randn(N,dim)
        # Set HMC params
        self._L_step_sz = 0.01
        self._L_accept_rate = 0.9

    @property
    def D(self):
        Mu = -((self.L[:,None,:] - self.L[None,:,:])**2).sum(2)
        Mu /= self.mu_0
        Mu += self.mu_self * np.eye(self.N)

        return Mu

    @property
    def P(self):
        P = logistic(self.D)
        return P

    def initialize_from_prior(self):
        self.mu_0 = np.random.randn()
        self.mu_self = np.random.randn()
        self.L = np.sqrt(self.sigma) * np.random.randn(self.N, self.dim)

    def synthetic_A(self):
        """
        Sample a new NxN network with the current distribution parameters
        :return:
        """
        P = self.P
        A = np.random.rand(self.N, self.N) < P

        return A

    def real_A(self,W):
        """
        Calculating A from data matrix W
        :return:
        """        
        sq = W*W
        return sq / (2*sq + 1 - 2*W) 