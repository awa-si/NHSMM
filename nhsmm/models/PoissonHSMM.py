# PoissonHSMM
from typing import Optional
import torch
from torch.distributions import Poisson, Independent

from nhsmm.hsmm.BaseHSMM import BaseHSMM
from nhsmm.utilities import utils


class PoissonHSMM(BaseHSMM):
    """
    Poisson Hidden Semi-Markov Model (HSMM)
    ----------
    Hidden Markov model with emissions. This model is a special case of the HSMM model with a geometric duration distribution.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int): 
        Number of emissions in the model.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial distribution, transition amd emission probabilities.
    seed (int):
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 max_duration:int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None):
        
        self.n_features = n_features
        super().__init__(n_states,max_duration,alpha,seed)

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1 + self.pdf.rates.numel()

    def sample_emission_pdf(self,X=None):
        if X is not None:
            rates = X.mean(dim=0).expand(self.n_states,-1).clone()
        else:
            rates = torch.ones(size=(self.n_states, self.n_features), 
                               dtype=torch.float64)

        return Independent(Poisson(rates),1)

    def _estimate_emission_pdf(self,X,posterior,theta=None):
        new_rates = self._compute_rates(X,posterior,theta)
        return Independent(Poisson(new_rates),1)

    def _compute_rates(self,
                       X:torch.Tensor,
                       posterior:torch.Tensor,
                       theta:Optional[utils.ContextualVariables]) -> torch.Tensor:
        """Compute the rates for each hidden state"""
        if theta is not None:
            # TODO: matmul shapes are inconsistent 
            raise NotImplementedError('Contextualized emissions not implemented for PoissonHMM')
        else:   
            new_rates = posterior.T @ X
            new_rates /= posterior.T.sum(1,keepdim=True)

        return new_rates