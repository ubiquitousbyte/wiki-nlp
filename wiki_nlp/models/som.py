from typing import Optional

import torch
import torch.nn as nn

import numpy as np
from math import log


class Decay(nn.Module):
    # An exponential decay function

    def __init__(self, sigma=1.0, time_constant=None):
        super(Decay, self).__init__()
        self._sigma = sigma
        self._time_constant = (time_constant if time_constant
                               else float(1000/log(self._sigma)))

    def forward(self, time_step):
        return self._sigma*np.exp(-time_step/self._time_constant)


class GaussianNeigh(nn.Module):

    def __init__(self, initial_sigma=1.0, time_constant=None):
        super(GaussianNeigh, self).__init__()
        self._decay = Decay(initial_sigma, time_constant=time_constant)

    def forward(self, x, y, winner, time_step):
        # Compute the lateral distances of the coordinates in the map
        # relative to the winner's coordinates
        distx = torch.pow(x - x.T[winner], 2)
        disty = torch.pow(y - y.T[winner], 2)

        # Compute the neighbourhood's normalizing constant
        # This represents the spread of the gaussian density
        # At each time step, the spread decreases.
        # This means that a smaller set of neighours will be updated.
        decay = self._decay.forward(time_step)

        # Map the distances onto the aforementioned gaussian density function
        # The neurons whose distances are smallest relative to the
        # winner are mapped around the peak of the bell curve and thus have high probabilities.
        hx = torch.exp(torch.neg(torch.div(distx, 2*decay*decay)))
        hy = torch.exp(torch.neg(torch.div(disty, 2*decay*decay)))
        h = (hx * hy).T

        return h


class QuantizationLoss(nn.Module):
    # Sums the distances between the nodes of a SOM and the data points.
    # The larger the sum, the more disproportional the mapping between
    # the input and the output space.

    def __init__(self):
        super(QuantizationLoss, self).__init__()

    def _weight_dist(self, x, w):
        wf = w.reshape(-1, w.size()[2])
        xsq = torch.sum(torch.pow(x, 2), dim=1, keepdim=True)
        wfsq = torch.sum(torch.pow(wf, 2), dim=1, keepdim=True)
        ct = torch.matmul(x, wf.T)
        return torch.sqrt(-2 * ct + xsq + wfsq.T)

    def _quantize(self, x, w):
        winners = torch.argmin(self._weight_dist(x, w), dim=1)
        return w[np.unravel_index(winners, w.size()[:2])]

    def forward(self, x, w):
        return torch.mean(torch.norm(x-self._quantize(x, w), dim=1))


class SOM(nn.Module):
    # A self-organizing map used for clustering and visualising document vectors
    # This implementation is inspired by the book
    # "Neural Networks and Learning Machines" 3rd. Edition by Simon Haykin
    # ISBN: 9780136097112

    def __init__(
        self,
        x_size: int,
        y_size: int,
        w_size: int,
        initial_sigma: float = 2.0,
        initial_alpha: float = 0.1,
        alpha_decay_const: float = 1000,
        time_constant: Optional[float] = None
    ):
        super(SOM, self).__init__()

        # The lattice containing the neurons that self-organize onto the document vectors
        # x_size defines the lattice's span across the x-axis
        # y_size defines the lattice's span across the y-axis
        # w_size defines the input size
        self._W = nn.Parameter(
            torch.randn(x_size, y_size, w_size), requires_grad=False
        )

        # The neighbourhood function used to determine a winner's neighbours
        self._neigh = GaussianNeigh(initial_sigma, time_constant)

        self._alpha_decay = Decay(initial_alpha, alpha_decay_const)

        # The 2-dimensional grid
        self._GX, self._GY = torch.meshgrid(
            torch.arange(x_size), torch.arange(y_size))

    def winner(self, x):
        d = torch.linalg.norm(x - self._W, ord=2, dim=-1)
        winner = np.unravel_index(torch.argmin(d), d.size())
        return winner

    def forward(self, x, time_step):
        # Competitive process
        # The winner neuron is the one which resembles the input the most
        # Find the winner by calculating the euclidean distance of all neurons to the input
        winner = self.winner(x)

        # Cooperative process
        # Compute a probability distribution over the discrete output space
        # that assigns high probabilities to the winner's neighbouring neurons
        h = self._neigh.forward(self._GX, self._GY, winner, time_step)

        return h, winner

    def backward(self, x, h, time_step):
        # Adaptive process
        # Update the winner neuron and its neighbours by pushing them closer to the input
        alpha = self._alpha_decay.forward(time_step)
        f = alpha * h
        self._W += torch.einsum('ij,ijk->ijk', f, x - self._W)

    def quantization_error(self, x):
        qloss = QuantizationLoss()
        return qloss.forward(x, self._W)
