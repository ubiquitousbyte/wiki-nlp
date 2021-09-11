import numpy as np 
from numpy.random import default_rng

from wiki_nlp.data.dataset import WikiDataset

class NoiseSampler:
    # The sampler allows us to avoid computing softmax's normalizing constant
    # which requires us to iterate over the whole vocabulary.
    # The idea is to replace the softmax function with a nonlinear logistic regression
    # that discriminates the observed data and some artifically generated noise.
    # The latter is, in our case, a set of randomly sampled words from a document. 
    # The artifically generated noise is sampled from a power law distribution 
    # that heuristically favours non-frequent words.
    # This object samples from the aforementioned distribution 

    def __init__(
        self, 
        dataset: WikiDataset, 
        noise_size: int
    ):
        self._noise_size = noise_size
        self._vocab = dataset.vocab
        self._counter = dataset.counter
        self._ps = np.zeros((len(self._vocab) -1),)
        for word, freq in self._counter.items():
            self._ps[self._vocab[word] - 1] = freq
        
        self._ps = np.power(self._ps, 0.75)
        self._ps /= np.sum(self._ps)
        self._rng = default_rng()

    def sample(self):
        return self._rng.choice(self._ps.shape[0], self._noise_size, p=self._ps).tolist()