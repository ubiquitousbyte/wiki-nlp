import torch
import torch.nn as nn


class DM(nn.Module):
    # An implementation of the Distributed Memory Model of Paragraph Vectors
    # proposed by Mikholov et. al. in 2014 in the paper
    # Distributed Representations of Sentences and Documents
    # https://arxiv.org/abs/1405.4053

    def __init__(self, embedding_dim: int, n_docs: int, n_words: int):
        super(DM, self).__init__()
        # Document matrix D -> [M x N]
        self._D = nn.Parameter(
            torch.randn(n_docs, embedding_dim), requires_grad=True
        )
        # Input word matrix W -> [V+1 x N]
        # The last row of the word matrix holds a sentinel vector that can be used
        # to map words that should not be updated during the training process
        # This is useful when combined with a word subsampler
        self._W = nn.Parameter(
            torch.cat((torch.randn(n_words, embedding_dim),
                      torch.zeros(1, embedding_dim))),
            requires_grad=True
        )
        # Output word matrix W' -> [N x V]
        self._Wp = nn.Parameter(
            torch.randn(embedding_dim, n_words), requires_grad=True
        )

    def forward(self, ctx_ids, doc_ids, target_and_noise_ids):
        # Step 1 (Input -> Hidden): Aggregate the context word vector batch with the document vector batch
        # Example:
        #   Let the document vector batch consist of 5 documents
        #   Let the context word vector batch consist of 5 contexts (one for each document),
        #   where each context consists of 10 context words
        #   Then, the executed computation is:
        #   [5 x N] + [5 x row_wise_sum([10 x N])] = [5 x N] + [5 x N] = [5 x N]
        #   Thus, we get the aggregated hidden state h for each of the 5 documents in the batch
        x = torch.add(self._D[doc_ids, :], torch.sum(
            self._W[ctx_ids, :], dim=1))

        # Step 2 (Hidden -> Output): Compute the similarity between (context, target) and (context, noise_samples)
        # This operation is implemented as a oneshot batch matrix multiplication
        # The only requirement is that the target index of the true center word is included in the noise batch
        # Example:
        #   Let the number of documents be 5
        #   Let the number of noise samples per document be 10.
        #   Thus, target_noise_ids has a size of [5 x 11] (This includes the prediction index of the actual center word \hat{y}).
        #   We already know that the hidden state has a size of [5 x N]
        #   We add an additional dimension for the batch computation, so [5 x N] becomes [5 x 1 x N]
        #   The executed computation is:
        #   [5 x 1 x N] x [N x 5 x 11] = [5 x 1 x 11]
        #   We remove the single dimension, so [5 x 1 x 11] becomes [5 x 11]
        #   Thus, we get the similarity between the hidden state, the negative samples and the true center word for every document
        return torch.bmm(
            x.unsqueeze(1),
            self._Wp[:, target_and_noise_ids].permute(1, 0, 2)).squeeze()
