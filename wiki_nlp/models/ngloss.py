import torch
import torch.nn as nn


class NegativeSamplingLoss(nn.Module):

    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        # k is the number of negative samples
        k = scores.size()[1] - 1
        # The first column in the score matrix holds the score of the true center word
        # All the other columns hold the probabilities of the negative samples
        # The loss is further normalized by the number of documents in the batch
        return -torch.sum(
            self._log_sigmoid(scores[:, 0]) +
            torch.sum(self._log_sigmoid(-scores[:, 1:]), dim=1) / k
        ) / scores.size()[0]
