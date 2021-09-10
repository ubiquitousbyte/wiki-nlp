import multiprocessing
import os 
import signal
from typing import Callable

from numpy import ceil
import torch

from wiki_nlp.models.noise_sampler import NoiseSampler
from wiki_nlp.data.dataset import (
    WikiDataset,
    WikiExample, 
    document_reader, 
    load_dataset
)

class _Batch(object): 
    # A batch of training examples.
    # The batch consists of a set of documents, 
    # the noise samples to use for the documents during training, 
    # as well as the context words to aggregate with each document. 

    def __init__(self):
        self.ctx_ids = []
        self.doc_ids = []
        self.tn_ids = []

    def __len__(self):
        return len(self.doc_ids)

    def torchify(self):
        self.ctx_ids = torch.LongTensor(self.ctx_ids)
        self.doc_ids = torch.LongTensor(self.doc_ids)
        self.tn_ids = torch.LongTensor(self.tn_ids)

    def cudify(self):
        self.ctx_ids = self.ctx_ids.cuda()
        self.doc_ids = self.doc_ids.cuda()
        self.tn_ids = self.tn_ids.cuda()

class _BatchState(object):

    def __init__(self, ctx_size: int):
        # The raw values are allocated in shared memory and
        # will be inherited by any child processes of the 
        # process instantiating this object. Coupled with a mutex, 
        # these values can be manipulated in parallel. 
        self._doc_id = multiprocessing.RawValue('i', 0)
        self._word_id = multiprocessing.RawValue('i', ctx_size)
        self._mutex = multiprocessing.Lock()

    def forward(
        self,
        dataset: WikiDataset,
        batch_size: int, 
        ctx_size: int,
        example_counter: Callable
    ):
        with self._mutex:
            doc_id, word_id = self._doc_id.value, self._word_id.value 
            self._forward(dataset, batch_size, ctx_size, example_counter)
            return doc_id, word_id 

    def _forward(
        self,
        dataset: WikiDataset,
        batch_size: int,
        ctx_size: int, 
        example_counter: Callable
    ):
        ex_count = example_counter(dataset[self._doc_id.value], self._word_id.value)

        if ex_count > batch_size:
            self._word_id.value += batch_size
            return 

        if ex_count == batch_size:
            if self._doc_id.value < len(dataset) - 1:
                self._doc_id.value += 1
            else:
                self._doc_id.value = 0
            self._word_id.value = ctx_size
            return 
        
        while ex_count < batch_size:
            if self._doc_id.value == len(dataset) - 1:
                self._doc_id.value = 0
                self._word_id.value = ctx_size
                return 
            
            self._doc_id.value += 1
            ex_count += example_counter(dataset[self._doc_id.value])
        
        self._word_id.value = (len(dataset[self._doc_id.value].text) 
                               - ctx_size 
                               - (ex_count - batch_size))


class _NoiseGenerator(object):

    def __init__(
        self, 
        dataset: WikiDataset, 
        batch_size: int, 
        ctx_size: int, 
        noise_size: int,
        state: _BatchState, 
    ):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.ctx_size = ctx_size
        self.noise_size = noise_size
        self.sampler = NoiseSampler(self.dataset, self.noise_size)
        self._vocab = self.dataset.vocab
        self._state = state 
    
    def forward(self):
        doc_id, word_id = self._state.forward(
            self.dataset,
            self.batch_size,
            self.ctx_size,
            self._example_counter
        )

        batch = _Batch()

        while len(batch) < self.batch_size:
            # Populate the batch 
            if doc_id == len(self.dataset):
                # All documents have been processed
                # Return the batch
                break 

            rem = len(self.dataset[doc_id].text) - 1 - self.ctx_size
            if word_id <= rem:
                # There are contexts yet to be processed
                self._populate_batch(doc_id, word_id, batch)
                word_id += 1
            else:
                # All contexts for this document have been processed
                # We therefore reset the word index and increment the document index 
                doc_id += 1
                word_id = self.ctx_size
        
        batch.torchify()
        return batch 

    def _populate_batch(self, doc_id: int, word_id: int, batch: _Batch):
        txt = self.dataset[doc_id].text

        # Add the document to the batch
        batch.doc_ids.append(doc_id)

        # Construct the document's noise samples
        # Make sure to include the true center word index in the noise at i=0
        noise = self.sampler.sample()
        noise.insert(0, self._stoi(txt[word_id]))
        batch.tn_ids.append(noise)

        # Construct the context
        ctx = []
        ctx_ids = (word_id + offset for offset in 
                   range(-self.ctx_size, self.ctx_size + 1) 
                   if offset != 0)
        for i in ctx_ids:
            ctx_id = self._stoi(txt[i])
            ctx.append(ctx_id)

        batch.ctx_ids.append(ctx)

    def _example_counter(self, example: WikiExample, word_id=None):
        if word_id is not None:
            # Count the fraction of unprocessed context words 
            # relative to the word index 
            if len(example.text) - word_id >= self.ctx_size + 1:
                return len(example.text) - word_id - self.ctx_size
            return 0
        
        if len(example.text) >= 2 * self.ctx_size + 1:
            return len(example.text) - 2 * self.ctx_size
        return 0 

    def _stoi(self, s: str) -> int:
        return self._vocab.get_stoi()[s]

    def __len__(self):
        examples = sum(self._example_counter(d) for d in self.dataset)
        return ceil(examples / self.batch_size)

class BatchGenerator(object):

    def __init__(
        self, 
        dataset: WikiDataset,
        batch_size: int, 
        ctx_size: int,
        noise_size: int, 
        max_size: int, 
        num_workers: int = multiprocessing.cpu_count()
    ):
        self.max_size = max_size 
        self.num_workers = num_workers

        self._noise_generator = _NoiseGenerator(
            dataset, 
            batch_size,
            ctx_size,
            noise_size,
            _BatchState(ctx_size),
        )
        self._queue = None
        self._stop_event = None  
        self._workers = []

    def __len__(self):
        return len(self._noise_generator)

    def start(self):
        self._queue = multiprocessing.Queue(maxsize=self.max_size)
        self._stop_event = multiprocessing.Event()

        for _ in range(self.num_workers):
            worker = multiprocessing.Process(target=self._work)
            worker.daemon = True 
            self._workers.append(worker)
            worker.start()
    
    def _work(self):
        while not self._stop_event.is_set():
            try:
                batch = self._noise_generator.forward()
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def __getstate__(self):
        # Python can't picke a list of processes, because a process
        # is not serializable. Took me ages to figure this one out. 
        state = self.__dict__.copy()
        state['_workers'] = None
        return state
    
    def stop(self):
        if self.is_running():
            self._stop_event.set()
        
        for worker in self._workers:
            if worker.is_alive():
                os.kill(worker.pid, signal.SIGINT)
                worker.join()
        
        if self._queue is not None:
            self._queue.close()
        
        self._queue = None 
        self._stop_event = None 
        self._workers = []

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def forward(self):
        while self.is_running():
            yield self._queue.get()

if __name__ == '__main__':
    ds = load_dataset(document_reader, start=0, end=64)
    batch_generator = BatchGenerator(
        ds, 
        batch_size=64, 
        ctx_size=3, 
        noise_size=10, 
        max_size=5,
        num_workers=8
    )
    batch_generator.start()
    for i in range(0, 2000):
        batch = next(batch_generator.forward())
        word = batch.tn_ids[0][0]
        print('Center word', ds.vocab.get_itos()[word])
        context = batch.ctx_ids[0]
        print([ds.vocab.get_itos()[c] for c in context])

    batch_generator.stop()
    
    