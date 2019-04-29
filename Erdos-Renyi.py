# "Stanard" Erdos-Renyi graph model and latent distance ER graph model" #

import abc

import numpy as np
from scipy.special import gammaln, psi
from scipy.misc import logsumexp

from pybasicbayes.abstractions import BayesianDistribution, GibbsSampling, MeanField, MeanFieldSVI
from pybasicbayes.util.stats import sample_discrete_from_log

from pyhawkes.internals.distributions import Discrete, Bernoulli, Gamma, Dirichlet, Beta

# Classes: ER and ER_Ldist #

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
    def __init__(self, K, p, kappa=1.0, alpha=None, beta=None, v=None, allow_self_connections=True):
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
    def __init__(self, K, dim=2,
                 v=None, alpha=1.0, beta=1.0,
                 kappa=1.0):
        super(LatentDistanceAdjacencyModel, self).\
            __init__(K=K, v=v, alpha=alpha, beta=beta, kappa=kappa)

        # Create a latent distance model for adjacency matrix
        from graphistician.adjacency import LatentDistanceAdjacencyDistribution
        self.A_dist = LatentDistanceAdjacencyDistribution(K, dim=dim)

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
