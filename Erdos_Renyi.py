# "Stanard" Erdos-Renyi graph model and latent distance ER graph model" #

import abc

import numpy as np
from scipy.special import gammaln, psi
from scipy.misc import logsumexp

from pybasicbayes.abstractions import BayesianDistribution, GibbsSampling, MeanField, MeanFieldSVI, Distribution
from pybasicbayes.util.stats import sample_discrete_from_log

from pyhawkes.internals.distributions import Discrete, Bernoulli, Gamma, Dirichlet, Beta
import pdb 

def logistic(x): 
    return 1./(1+np.exp(-x))

def get_W(data):
    dim = len(data)
    W = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1,dim):
            W[i,j] = np.linalg.norm(data[i] - data[j])
    
    W += W.T
    W_norm = 1/np.linalg.norm(W) * W 

    return W_norm

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
                 dim=2, sigma=1.0, mu0=0.0, mu_self=0.0):
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
        Mu += self.mu_0
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
        logit_P = D + mu_0 + mu_self * np.eye(self.N)

        # Take the logistic of the negative distance
        P = 1.0 / (1+anp.exp(-logit_P))

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

class ErdosRenyiFixedSparsity(GibbsSampling, MeanField):
    """
    An Erdos-Renyi model with "constant" sparsity rho:

    K:   Number of nodes
    v:   Scale of the gamma weight distribution from node to node
    p:   Sparsity of the network
    kappa: Weight matrix parameter..
    
    Parameters (Bayes):
    alpha   Shape parameter of gamma prior over v
    beta    Scale parameter of gamma prior over v
    """
    def __init__(self, K, p, L, kappa=1.0, alpha=None, beta=None, v=None, allow_self_connections=True):
        self.K = K
        self.p = p
        self.kappa = kappa
        
        # Set the weight scale
        if alpha is beta is v is None:
            # If no parameters are specified, set v to be as large as possible
            # while still being stable with high probability
            # See the original paper for details
            self.v = K * kappa * p / 0.5
            self.alpha = self.beta = None
        elif v is not None:
            self.v = v
            self.alpha = self.beta = None
        elif alpha is not None:
            self.alpha = alpha
            if beta is not None:
                self.beta= beta
            else:
                self.beta = alpha * K
            self.v = self.alpha / self.beta
        else:
            raise NotImplementedError("Invalid v,alpha,beta settings")


        self.allow_self_connections = allow_self_connections

        # Mean field
        if self.alpha and self.beta:
            self.mf_alpha = self.alpha
            self.mf_beta = self.beta

    @property
    def P(self):
        """
        Get the KxK matrix of probabilities
        :return:
        """
        P = self.p * np.ones((self.K, self.K))
        if not self.allow_self_connections:
            np.fill_diagonal(P, 0.0)
        return P

    @property
    def V(self):
        """
        Get the KxK matrix of scales
        :return:
        """
        return self.v * np.ones((self.K, self.K))

    @property
    def Kappa(self):
        return self.kappa * np.ones((self.K, self.K))

    def log_likelihood(self, x):
        """
        Compute the log likelihood of a set of SBM parameters

        :param x:    (m,p,v) tuple
        :return:
        """
        lp = 0
        lp += Gamma(self.alpha, self.beta).log_probability(self.v).sum()
        return lp

    def log_probability(self):
        return self.log_likelihood((self.m, self.p, self.v, self.c))

    def rvs(self,size=[]):
        raise NotImplementedError()

    ### Gibbs sampling
    def resample_v(self, A, W):
        """
        Resample v given observations of the weights
        """
        alpha = self.alpha + A.sum() * self.kappa
        beta  = self.beta + W[A > 0].sum()
        self.v = np.random.gamma(alpha, 1.0/beta)

    def resample(self, data=[]):
        if all([self.alpha, self.beta]):
            A,W = data
            self.resample_v(A, W)

    ### Mean Field
    def expected_p(self):
        return self.P

    def expected_notp(self):
        return 1.0 - self.expected_p()

    def expected_log_p(self):
        return np.log(self.P)

    def expected_log_notp(self):
         return np.log(1.0 - self.P)

    def expected_v(self):
        E_v = self.mf_alpha / self.mf_beta
        return E_v

    def expected_log_v(self):
        return psi(self.mf_alpha) - np.log(self.mf_beta)

    def expected_log_likelihood(self,x):
        pass

    def mf_update_v(self, E_A, E_W_given_A, stepsize=1.0):
        """
        Mean field update for the CxC matrix of block connection scales
        :param E_A:
        :param E_W_given_A: Expected W given A
        :return:
        """
        alpha_hat = self.alpha + (E_A * self.kappa).sum()
        beta_hat  = self.beta + (E_A * E_W_given_A).sum()
        self.mf_alpha = (1.0 - stepsize) * self.mf_alpha + stepsize * alpha_hat
        self.mf_beta  = (1.0 - stepsize) * self.mf_beta + stepsize * beta_hat

    def meanfieldupdate(self, weight_model, stepsize=1.0):
        E_A = weight_model.expected_A()
        E_W_given_A = weight_model.expected_W_given_A(1.0)
        self.mf_update_v(E_A=E_A, E_W_given_A=E_W_given_A, stepsize=stepsize)

    def meanfield_sgdstep(self,weight_model, minibatchfrac, stepsize):
        self.meanfieldupdate(weight_model, stepsize)

    def get_vlb(self):
        vlb = 0
        vlb += Gamma(self.alpha, self.beta).\
            negentropy(E_lambda=self.mf_alpha/self.mf_beta,
                       E_ln_lambda=psi(self.mf_alpha) - np.log(self.mf_beta)).sum()

        # Subtract the negative entropy of q(v)
        vlb -= Gamma(self.mf_alpha, self.mf_beta).negentropy().sum()
        return vlb

    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        self.v = np.random.gamma(self.mf_alpha, 1.0/self.mf_beta)

class LatentDistanceAdjacencyModel(ErdosRenyiFixedSparsity):
    """
    Network model with probability of connection given by
    a latent distance model. Depends on the graphistician package..
    """
    def __init__(self, K, p, L, dim=2,
                 v=None, alpha=1.0, beta=1.0,
                 kappa=1.0):
        super(LatentDistanceAdjacencyModel, self).\
            __init__(K=K, p = p, L = L, v=v, alpha=alpha, beta=beta, kappa=kappa)

        # Create a latent distance model for adjacency matrix
        self.A_dist = LatentDistanceAdjacencyDistribution(N=K, L=L, dim=dim)
        
    @property
    def P(self):
        return self.A_dist.P

    @property
    def L(self):
        return self.A_dist.L

    def resample(self, data=[]):
        A,W = data
        self.resample_v(A, W)
        self.A_dist.resample(A)


class SpikeAndSlabGammaWeights(GibbsSampling):
    """
    Encapsulates the KxK Bernoulli adjacency matrix and the
    KxK gamma weight matrix. Implements Gibbs sampling given
    the parent variables.
    """
    def __init__(self, model, parallel_resampling=True):
        """
        Initialize the spike-and-slab gamma weight model with either a
        network object containing the prior or rho, alpha, and beta to
        define an independent model.
        """
        self.model = None
        self.K = model.K
        # assert isinstance(network, GibbsNetwork), "network must be a GibbsNetwork object"
        self.network = model

        # Specify whether or not to resample the columns of A in parallel
        self.parallel_resampling = parallel_resampling

        # Initialize parameters A and W
        self.A = np.ones((self.K, self.K))
        self.W = np.zeros((self.K, self.K))
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
        assert isinstance(A, np.ndarray) and A.shape == (self.K,self.K), \
            "A must be a KxK adjacency matrix"
        assert isinstance(W, np.ndarray) and W.shape == (self.K,self.K), \
            "W must be a KxK weight matrix"

        # LL of A
        rho = np.clip(self.network.P, 1e-32, 1-1e-32)
        ll = (A * np.log(rho) + (1-A) * np.log(1-rho)).sum()
        ll = np.nan_to_num(ll)

        # Get the shape and scale parameters from the network model
        kappa = self.network.kappa
        v = self.network.V

        # Add the LL of the gamma weights
        lp_W = kappa * np.log(v) - gammaln(kappa) + \
               (kappa-1) * np.log(W) - v * W
        ll += (A*lp_W).sum()

        return ll

    def log_probability(self):
        return self.log_likelihood((self.A, self.W))

    def rvs(self,size=[]):
        A = np.random.rand(self.K, self.K) < self.network.P
        W = np.random.gamma(self.network.kappa, 1.0/self.network.V,
                            size(self.K, self.K))

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
            self.A = np.random.rand(self.K, self.K) < self.network.P
            return

        # We can naively parallelize over receiving neurons, k2
        # To avoid serializing and copying the data object, we
        # manually extract the required arrays Sk, Fk, etc.
        A_cols = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(par._resample_column_of_A)(k2)for k2 in range(self.K))
        self.A = np.array(A_cols).T

    def _resample_A_given_W(self, data):
        """
        Resample A given W. This must be immediately followed by an
        update of z | A, W.
        :return:
        """
        p = self.network.P
        for k1 in range(self.K):
            for k2 in range(self.K):
                if self.model is None:
                    ll0 = 0
                    ll1 = 0
                else:
                    # Compute the log likelihood of the events given W and A=0
                    self.A[k1,k2] = 0
                    ll0 = self.log_likelihood([self.A,data[1]]) #sum([d.log_likelihood_single_process(k2) for d in data])

                    # Compute the log likelihood of the events given W and A=1
                    self.A[k1,k2] = 1
                    ll1 = self.log_likelihood([self.A,data[1]]) #sum([d.log_likelihood_single_process(k2) for d in data])

                # Sample A given conditional probability
                lp0 = ll0 + np.log(1.0 - p[k1,k2])
                lp1 = ll1 + np.log(p[k1,k2])
                Z   = logsumexp([lp0, lp1])

                # ln p(A=1) = ln (exp(lp1) / (exp(lp0) + exp(lp1)))
                #           = lp1 - ln(exp(lp0) + exp(lp1))
                #           = lp1 - Z
                self.A[k1,k2] = np.log(np.random.rand()) < lp1 - Z
        
    def resample_new(self,data=[]):
        self._resample_A_given_W(data)
        #pdb.set_trace()
        self.resample_W_given_A_and_z()


    def resample_W_given_A_and_z(self, data=[]):
        """
        Resample the weights given A and z.
        :return:
        """
        #import pdb; pdb.set_trace()

        ss = np.zeros((2, self.K, self.K)) # + \
        #     sum([d.compute_weight_ss() for d in data])

        # Account for whether or not a connection is present in N
        ss[1] *= self.A

        kappa_post = self.network.kappa + ss[0]
        v_post  = self.network.V + ss[1 ]

        self.W = np.atleast_1d(np.random.gamma(kappa_post, 1.0/v_post)).reshape((self.K, self.K))

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
