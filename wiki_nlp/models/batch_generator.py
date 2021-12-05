import multiprocessing
import os
import signal
from typing import Callable

import torch
from math import ceil

from wiki_nlp.models.noise_sampler import NoiseSampler
from wiki_nlp.models.word_sampler import WordSampler

from wiki_nlp.data.dataset import (
    WikiDataset,
    WikiExample,
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
        # process instantiating this object.
        # Coupled with a mutex, these values can be manipulated concurrently
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
        ex_count = example_counter(
            dataset[self._doc_id.value], self._word_id.value)

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
        self.noise_sampler = NoiseSampler(self.dataset, self.noise_size)
        self.word_sampler = WordSampler(self.dataset)
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
                # Recompute the sampling probabilities and return the batch
                self.word_sampler.recompute_probs()
                break

            rem = len(self.dataset[doc_id].text) - 1 - self.ctx_size
            if word_id <= rem:
               # Check if the current center word has a high enough probability of being sampled
                if self.word_sampler.use_word(word_id):
                    # There are contexts in the current document that are yet to be processed
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
        noise = self.noise_sampler.sample()
        noise.insert(0, self._stoi(txt[word_id]))
        batch.tn_ids.append(noise)

        # Construct the context
        ctx = []
        ctx_ids = (word_id + offset for offset in
                   range(-self.ctx_size, self.ctx_size + 1)
                   if offset != 0)
        for i in ctx_ids:
            ctx_id = self._stoi(txt[i])
            if self.word_sampler.use_word(ctx_id):
                ctx.append(ctx_id)
            else:
                ctx.append(self._vocab.get_default_index())

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
        return self._vocab[s]

    def __len__(self):
        examples = sum(self._example_counter(d) for d in self.dataset)
        return ceil(examples / self.batch_size)


class BatchGenerator(object):
    # A concurrent batch generator

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
        # Starts the batch generator
        # This function spawns a set of workers, each tasked with
        # creating batches of input data to feed in the paragraph-vector model.
        # The workers store those batches in a process-safe event queue
        # The training loop shall sample batches from the queue.

        self._queue = multiprocessing.Queue(maxsize=self.max_size)
        self._stop_event = multiprocessing.Event()

        for _ in range(self.num_workers):
            worker = multiprocessing.Process(target=self._work)
            worker.daemon = True
            self._workers.append(worker)
            worker.start()

    def _work(self):
        # The worker loop that generates batches and puts them in the queue
        while not self._stop_event.is_set():
            try:
                batch = self._noise_generator.forward()
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def __getstate__(self):
        # Python can't pickle a list of processes, because a process
        # is not serializable. Took me ages to figure this one out.
        state = self.__dict__.copy()
        state['_workers'] = None
        return state

    def stop(self):
        # Stops the batch generator by killing all workers
        # and closing the queue

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
        # The API used by the training algorithm to sample batches
        # This function pops a batch from the queue and passes it to the caller
        while self.is_running():
            yield self._queue.get()
