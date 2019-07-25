import abc
import numpy as np
from scipy.special import gammaln, psi
from scipy.misc import logsumexp
from pybasicbayes.abstractions import BayesianDistribution, GibbsSampling, MeanField, MeanFieldSVI, Distribution
from pybasicbayes.util.stats import sample_discrete_from_log
from pyhawkes.internals.distributions import Discrete, Bernoulli, Gamma, Dirichlet, Beta
import pdb 
from scipy.spatial.distance import cosine 
import scipy as sp
from matplotlib import pyplot as plt 
import statsmodels.api as sm

def logistic(x): 
    return np.exp(x) #1./(1+np.exp(x))

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

def get_W_real(W,threshold=15):
    # get distribution from W 
    K = 30
    N = W.shape[0]
    # Remove zeros on diagonals from matrix
    W_no_diag = np.setdiff1d(np.concatenate(W),[0.])
    # Univariate KDE with Gaussian
    kde = sm.nonparametric.KDEUnivariate(W_no_diag)
    kde.fit() 

    # Printing the densitry fit to histogram
    #print("Value threshold: %s" % np.percentile(kde.support,q=threshold))
    #fig, ax = plt.subplots(figsize=(8, 6))
    #ax.hist(W_no_diag, bins=K, density=True, lw=0, alpha=0.5)
    #ax.plot(kde.support, kde.density, lw=3, label='Kernel Density Estimation', zorder=10)
    #plt.show()

    # make binary matrix A from W
    A = W < np.percentile(kde.support,q=threshold)
    A = A.astype(int)
    W_real = A*W
    
    return W_real

# Classes: ER and ER_Ldist #
class AdjacencyDistribution(Distribution):
    """
    Base class for a distribution over adjacency matrices.
    Must expose a matrix of connection probabilities.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, N):
        self.N = N

    @abc.abstractproperty
    def P(self):
        """
        :return: An NxN matrix of connection probabilities.
        """
        return np.nan

    @property
    def safe_P(self):
        return np.clip(self.P, 1e-64, 1-1e-64)

    @property
    def is_deterministic(self):
        """
        Are all the entries in P either 0 or 1?
        """
        P = self.P
        return np.all(np.isclose(P,0) | np.isclose(P,1))

    @abc.abstractmethod
    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return np.nan

    def log_likelihood(self, A):
        assert A.shape == (self.N, self.N)

        P = self.safe_P
        return np.sum(A * np.log(self.P) + (1-A) * np.log(1-P))

    def log_probability(self, A):
        return self.log_likelihood(A) + self.log_prior()

    def rvs(self,size=[]):
        # TODO: Handle size
        P = self.P
        return np.random.rand(*P.shape) < P

    @abc.abstractmethod
    def initialize_from_prior(self):
        raise NotImplementedError

    def initialize_hypers(self, A):
        pass

    @abc.abstractmethod
    def sample_predictive_parameters(self):
        """
        Sample a predictive set of parameters for a new row and column of A
        :return Prow, Pcol, each an N+1 vector. By convention, the last entry
                is the new node.
        """
        raise NotImplementedError

    def sample_predictive_distribution(self):
        """
        Sample a new row and column of A
        """
        N = self.N
        Prow, Pcol = self.sample_predictive_parameters()
        # Make sure they are consistent in the (N+1)-th entry
        assert Prow[-1] == Pcol[-1]

        # Sample and make sure they are consistent in the (N+1)-th entry
        Arow = np.random.rand(N+1) < Prow
        Acol = np.random.rand(N+1) < Pcol
        Acol[-1] = Arow[-1]

        return Arow, Acol

    def approx_predictive_ll(self, Arow, Acol, M=100):
        """
        Approximate the (marginal) predictive probability by averaging over M
        samples of the predictive parameters
        """
        N = self.N
        assert Arow.shape == Acol.shape == (N+1,)
        Acol = Acol[:-1]

        # Get the predictive parameters
        lps = np.zeros(M)
        for m in xrange(M):
            Prow, Pcol = self.sample_predictive_parameters()
            Prow = np.clip(Prow, 1e-64, 1-1e-64)
            Pcol = np.clip(Pcol, 1e-64, 1-1e-64)

            # Only use the first N entries of Pcol to avoid double counting
            Pcol = Pcol[:-1]

            # Compute lp
            lps[m] += (Arow * np.log(Prow) + (1-Arow) * np.log(1-Prow)).sum()
            lps[m] += (Acol * np.log(Pcol) + (1-Acol) * np.log(1-Pcol)).sum()

        # Compute average log probability
        lp = -np.log(M) + logsumexp(lps)
        return lp

class LatentDistanceAdjacencyDistribution(AdjacencyDistribution, GibbsSampling):
    
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
        #self.L = np.sqrt(self.sigma) * np.random.randn(self.N, self.dim)

    def initialize_hypers(self, A):
        pass

    def log_prior(self):
        """
        Compute the prior probability of F, mu0, and lmbda
        """
        lp  = 0

        # Log prior of F under spherical Gaussian prior
        from scipy.stats import norm
        lp += norm.logpdf(self.L, 0, np.sqrt(self.sigma)).sum()

        # Log prior of mu_0 and mu_self
        lp += norm.logpdf(self.mu_0, 0, 1)
        lp += norm.logpdf(self.mu_self, 0, 1)
        return lp

    def _hmc_log_probability(self, L, mu_0, mu_self, A):
        """
        Compute the log probability as a function of L.
        This allows us to take the gradients wrt L using autograd.
        :param L:
        :param A:
        :return:
        """
        import autograd.numpy as anp
        # Compute pairwise distance
        L1 = anp.reshape(L,(self.N,1,self.dim))
        L2 = anp.reshape(L,(1,self.N,self.dim))
        D = - anp.sum((L1-L2)**2, axis=2)

        # Compute the logit probability
        logit_P = D / mu_0 + mu_self * np.eye(self.N)

        # Take the logistic of the negative distance
        P = anp.exp(logit_P) #1.0 / (1+anp.exp(-logit_P))

        # Compute the log likelihood
        ll = anp.sum(A * anp.log(P) + (1-A) * anp.log(1-P))

        # Log prior of L under spherical Gaussian prior
        lp = -0.5 * anp.sum(L * L / self.sigma)

        # Log prior of mu0 under standardGaussian prior
        lp += -0.5 * mu_0**2

        lp += -0.5 * mu_self**2

        return ll + lp


    def rvs(self, size=[]):
        """
        Sample a new NxN network with the current distribution parameters
        :param size:
        :return:
        """
        # TODO: Sample the specified number of graphs
        P = self.P
        A = np.random.rand(self.N, self.N) < P

        return A

    def sample_predictive_parameters(self):
        Lext = \
            np.vstack((self.L, np.sqrt(self.sigma) * np.random.randn(1, self.dim)))

        D = -((Lext[:,None,:] - Lext[None,:,:])**2).sum(2)
        D += self.mu_0
        D += self.mu_self * np.eye(self.N+1)

        P = logistic(D)
        Prow = P[-1,:]
        Pcol = P[:,-1]

        return Prow, Pcol

    def plot(self, A, ax=None, color='k', L_true=None, lmbda_true=None):
        """
        If D==2, plot the embedded nodes and the connections between them
        :param L_true:  If given, rotate the inferred features to match F_true
        :return:
        """

        import matplotlib.pyplot as plt

        assert self.dim==2, "Can only plot for D==2"

        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, aspect="equal")

        # If true locations are given, rotate L to match L_true
        L = self.L
        if L_true is not None:
            from graphistician.internals.utils import compute_optimal_rotation
            R = compute_optimal_rotation(self.L, L_true)
            L = L.dot(R)

        # Scatter plot the node embeddings
        ax.plot(L[:,0], L[:,1], 's', color=color, markerfacecolor=color, markeredgecolor=color)
        # Plot the edges between nodes
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                if A[n1,n2]:
                    ax.plot([L[n1,0], L[n2,0]],
                            [L[n1,1], L[n2,1]],
                            '-', color=color, lw=1.0)

        # Get extreme feature values
        b = np.amax(abs(L)) + L[:].std() / 2.0

        # Plot grids for origin
        ax.plot([0,0], [-b,b], ':k', lw=0.5)
        ax.plot([-b,b], [0,0], ':k', lw=0.5)

        # Set the limits
        ax.set_xlim([-b,b])
        ax.set_ylim([-b,b])

        # Labels
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        plt.show()

        return ax

    def resample(self, A):
        """
        Resample the parameters of the distribution given the observed graphs.
        :param data:
        :return:
        """
        # Sample the latent positions
        self._resample_L(A)

        # Resample the offsets
        self._resample_mu_0(A)
        self._resample_mu_self(A)
        self._resample_sigma()

    def _resample_L(self, A):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc

        lp  = lambda L: self._hmc_log_probability(L, self.mu_0, self.mu_self, A)
        dlp = grad(lp)

        nsteps = 10
        self.L, self._L_step_sz, self._L_accept_rate = \
            hmc(lp, dlp, self._L_step_sz, nsteps, self.L.copy(),
                negative_log_prob=False, avg_accept_rate=self._L_accept_rate,
                adaptive_step_sz=True)

        # print "Var L: ", np.var(self.L)

    def _resample_mu_0(self, A):
        """
        Resample the locations given A
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc


        lp  = lambda mu_0: self._hmc_log_probability(self.L, mu_0, self.mu_self, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        mu_0 = hmc(lp, dlp, stepsz, nsteps, np.array(self.mu_0), negative_log_prob=False)
        self.mu_0 = float(mu_0)

    def _resample_mu_self(self, A):
        """
        Resample the self connection offset
        :return:
        """
        from autograd import grad
        from hips.inference.hmc import hmc


        lp  = lambda mu_self: self._hmc_log_probability(self.L, self.mu_0, mu_self, A)
        dlp = grad(lp)

        stepsz = 0.005
        nsteps = 10
        mu_self = hmc(lp, dlp, stepsz, nsteps, np.array(self.mu_self), negative_log_prob=False)
        self.mu_self = float(mu_self)

    def _resample_sigma(self):
        """
        Resample sigma under an inverse gamma prior, sigma ~ IG(1,1)
        :return:
        """
        L = self.L

        a_prior = 1.0
        b_prior = 1.0

        a_post = a_prior + L.size / 2.0
        b_post = b_prior + (L**2).sum() / 2.0

        from scipy.stats import invgamma
        self.sigma = invgamma.rvs(a=a_post, scale=b_post)

class SpikeAndSlabGammaWeights(GibbsSampling):
    """
    Encapsulates the KxK Bernoulli adjacency matrix and the
    KxK gamma weight matrix. Implements Gibbs sampling given
    the parent variables.
    """
    def __init__(self, distribution, parallel_resampling=True, kappa=1.00, v=1.00):
        """
        Initialize the spike-and-slab gamma weight model with either a
        network object containing the prior or rho, alpha, and beta to
        define an independent model.
        """
        self.v = v
        self.kappa = kappa
        self.distribution = None
        self.N = distribution.N
        # assert isinstance(network, GibbsNetwork), "network must be a GibbsNetwork object"
        self.distribution = distribution

        # Specify whether or not to resample the columns of A in parallel
        self.parallel_resampling = parallel_resampling

        # Initialize parameters A and W
        self.A = np.ones((self.N, self.N))
        self.W = np.zeros((self.N, self.N))
        #self.resample()

    @property
    def W_effective(self):
        return self.A * self.W

    def log_likelihood(self, x):
        """
        Compute the log likelihood of the given A and W
        :param x:  an (A,W) tuple
        :return:
        """
        A,W = x
        assert isinstance(A, np.ndarray) and A.shape == (self.N,self.N), \
            "A must be a KxK adjacency matrix"
        assert isinstance(W, np.ndarray) and W.shape == (self.N,self.N), \
            "W must be a KxK weight matrix"

        # LL of A
        rho = np.clip(self.distribution.P, 1e-32, 1-1e-32)
        ll = (A * np.log(rho) + (1-A) * np.log(1-rho)).sum()
        ll = np.nan_to_num(ll)

        # Get the shape and scale parameters from the network model
        kappa = self.kappa
        v = self.v

        # Add the LL of the gamma weights

        log_W = np.log(W)
        log_W[np.isinf(log_W)] = 0 
        lp_W = kappa * np.log(v) - gammaln(kappa) + (kappa-1) * log_W - v * W
        ll += (A*lp_W).sum()

        return ll

    def log_probability(self):
        return self.log_likelihood((self.A, self.W))

    def rvs(self,size=[]):
        A = np.random.rand(self.N, self.N) < self.distribution.P
        W = np.random.gamma(self.kappa, 1.0/self.v,
                            size(self.N, self.N))

        return A,W

    def _joint_resample_A_W(self):
        """
        Not sure how to do this yet, but it would be nice to resample A
        from its marginal distribution after integrating out W, and then
        sample W | A.
        :return:
        """
        raise NotImplementedError()

    def _joblib_resample_A_given_W(self, data):
        """
        Resample A given W. This must be immediately followed by an
        update of z | A, W. This  version uses joblib to parallelize
        over columns of A.
        :return:
        """
        # Use the module trick to avoid copying globals
        import pyhawkes.internals.parallel_adjacency_resampling as par
        par.model = self.model
        par.data = data
        par.K = self.model.K

        if len(data) == 0:
            self.A = np.random.rand(self.N, self.N) < self.network.P
            return

        # We can naively parallelize over receiving neurons, k2
        # To avoid serializing and copying the data object, we
        # manually extract the required arrays Sk, Fk, etc.
        A_cols = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(par._resample_column_of_A)(k2)for k2 in range(self.K))
        self.A = np.array(A_cols).T
        pass
        #related to Hawkes, so we don't need it here
    def _resample_A_given_W(self, data):
        """
        Resample A given W. This must be immediately followed by an
        update of z | A, W.
        :return:
        """
        p = self.distribution.P
        for n1 in range(self.N):
            for n2 in range(self.N):
                if self.distribution is None:
                    ll0 = 0
                    ll1 = 0
                else:
                    # Compute the log likelihood of the events given W and A=0
                    self.A[n1,n2] = 0
                    ll0 = self.log_likelihood([self.A,data[1]]) #sum([d.log_likelihood_single_process(k2) for d in data])

                    # Compute the log likelihood of the events given W and A=1
                    self.A[n1,n2] = 1
                    ll1 = self.log_likelihood([self.A,data[1]]) #sum([d.log_likelihood_single_process(k2) for d in data])

                # Apply Bayes in a weird way 
                # Sample A given conditional probability
                if p[n1,n2] == 1:
                    lp1 = 1
                    lp0 = 0
                elif p[n1,n2] == 0:
                    lp1 = 0
                    lp0 = 1
                else: 
                    lp0 = ll0 + np.log(1.0 - p[n1,n2])
                    lp1 = ll1 + np.log(p[n1,n2])
                Z   = logsumexp([lp0, lp1])
                f_aij = lp1 - Z
                G = np.random.gamma(self.kappa, 1.0)
                # ln p(A=1) = ln (exp(lp1) / (exp(lp0) + exp(lp1)))
                #           = lp1 - ln(exp(lp0) + exp(lp1))
                #           = lp1 - Z
                self.A[n1,n2] = np.log(np.random.rand()) < f_aij/G 
        
    def resample_new(self,data=[]):
        self._resample_A_given_W(data)
        #pdb.set_trace()
        self.resample_W_given_A_and_z()

    #Hawkes...
    def resample_W_given_A_and_z(self, data=[]):
        """
        Resample the weights given A and z.
        :return:
        """
        #import pdb; pdb.set_trace()

        ss = np.zeros((2, self.N, self.N)) # + \
        #     sum([d.compute_weight_ss() for d in data])

        # Account for whether or not a connection is present in N
        ss[1] *= self.A

        kappa_post = self.kappa + ss[0]
        #v_post  = self.network.V + ss[1 ]

        self.W = np.atleast_1d(np.random.gamma(kappa_post, 1.0)).reshape((self.N, self.N))
        pass

    def resample(self, data=[]):
        """
        Resample A and W given the parents
        :param N:   A length-K vector specifying how many events occurred
                    on each of the K processes
        :param Z:   A TxKxKxB array of parent assignment counts
        """
        # Resample W | A
        self.resample_W_given_A_and_z(data)

        # Resample A given W
        if self.parallel_resampling:
            self._joblib_resample_A_given_W(data)
        else:
            self._resample_A_given_W(data)


