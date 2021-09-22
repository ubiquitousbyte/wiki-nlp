import numpy as np

from wiki_nlp.data.dataset import WikiDataset

class WordSampler:
    # The word sampler assings probabilities to every word in the vocabulary 
    # Overfrequent words are assigned low probabilities.
    # Conversely, infrequent wrods are assigned high probabilities.
    # This is a heuristic approach that attempts to push words towards
    # meaningful contexts and push them away from ubiquitous words such as "der", "die", "das" etc. 

    def __init__(self, dataset: WikiDataset, sampling_rate: float = 0.001):
        self.dataset = dataset 
        self.word_count = len(self.dataset.vocab)
        self.sampling_rate = sampling_rate
        self._use_mask = []
        self.recompute_probs()

    def recompute_probs(self):
        self._use_mask = []
        for word, freq in self.dataset.counter.items():
            f = freq/self.word_count
            p = (np.sqrt(f/self.sampling_rate)+1)*(self.sampling_rate/f)
            self._use_mask.insert(self.dataset.vocab[word], p > np.random.random_sample())

    def use_word(self, word_idx: int) -> bool:
        return self._use_mask[word_idx]