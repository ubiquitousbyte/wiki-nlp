import time
from unittest import TestCase

from wiki_nlp.data.dataset import load_dataset, document_reader
from wiki_nlp.models.batch_generator import BatchGenerator

class GeneratorTest(TestCase):

    def setUp(self):
        self.dataset = load_dataset(document_reader, end=16)

    def test_num_examples_for_different_batch_sizes(self):
        len_1 = self._num_examples_with_batch_size(1)

        for batch_size in range(2, 5):
            len_x = self._num_examples_with_batch_size(batch_size)
            self.assertEqual(len_x, len_1)

    def _num_examples_with_batch_size(self, batch_size):
        generator = BatchGenerator(
            self.dataset,
            batch_size=batch_size,
            ctx_size=2,
            noise_size=3,
            max_size=1,
            num_workers=1)

        num_batches = len(generator)
        generator.start()
        nce_generator = generator.forward()

        total = 0
        for _ in range(num_batches):
            batch = next(nce_generator)
            total += len(batch)
        generator.stop()
        return total

    def test_multiple_iterations(self):
        nce_data = BatchGenerator(
            self.dataset,
            batch_size=16,
            ctx_size=3,
            noise_size=3,
            max_size=1,
            num_workers=1)
        num_batches = len(nce_data)
        nce_data.start()
        nce_generator = nce_data.forward()

        iter0_targets = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            iter0_targets.append([x[0] for x in batch.tn_ids])

        iter1_targets = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            iter1_targets.append([x[0] for x in batch.tn_ids])

        for ts0, ts1 in zip(iter0_targets, iter1_targets):
            for t0, t1 in zip(ts0, ts0):
                self.assertEqual(t0, t1)
        nce_data.stop()

    def test_different_batch_sizes(self):
        nce_data = BatchGenerator(
            self.dataset,
            batch_size=16,
            ctx_size=1,
            noise_size=3,
            max_size=1,
            num_workers=1)
        num_batches = len(nce_data)
        nce_data.start()
        nce_generator = nce_data.forward()

        targets0 = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            for ts in batch.tn_ids:
                targets0.append(ts[0])
        nce_data.stop()

        nce_data = BatchGenerator(
            self.dataset,
            batch_size=19,
            ctx_size=1,
            noise_size=3,
            max_size=1,
            num_workers=1)
        num_batches = len(nce_data)
        nce_data.start()
        nce_generator = nce_data.forward()

        targets1 = []
        for _ in range(num_batches):
            batch = next(nce_generator)
            for ts in batch.tn_ids:
                targets1.append(ts[0])
        nce_data.stop()

        for t0, t1 in zip(targets0, targets1):
            self.assertEqual(t0, t1)


    def test_parallel(self):
        # serial version has max_size=3, because in the parallel version two
        # processes advance the state before they are blocked by the queue.put()
        nce_data = BatchGenerator(
            self.dataset,
            batch_size=32,
            ctx_size=5,
            noise_size=1,
            max_size=3,
            num_workers=1)
        nce_data.start()
        time.sleep(1)
        nce_data.stop()
        state_serial = nce_data._noise_generator._state

        nce_data = BatchGenerator(
            self.dataset,
            batch_size=32,
            ctx_size=5,
            noise_size=1,
            max_size=2,
            num_workers=2)
        nce_data.start()
        time.sleep(1)
        nce_data.stop()
        state_parallel = nce_data._noise_generator._state

        self.assertEqual(
            state_parallel._doc_id.value,
            state_serial._doc_id.value)
        self.assertEqual(
            state_parallel._word_id.value,
            state_serial._word_id.value)

    def test_tensor_sizes(self):
        nce_data = BatchGenerator(
            self.dataset,
            batch_size=32,
            ctx_size=5,
            noise_size=3,
            max_size=1,
            num_workers=1)
        nce_data.start()
        nce_generator = nce_data.forward()
        batch = next(nce_generator)
        nce_data.stop()

        self.assertEqual(batch.ctx_ids.size()[0], 32)
        self.assertEqual(batch.ctx_ids.size()[1], 10)
        self.assertEqual(batch.doc_ids.size()[0], 32)
        self.assertEqual(batch.tn_ids.size()[0], 32)
        self.assertEqual(batch.tn_ids.size()[1], 4)