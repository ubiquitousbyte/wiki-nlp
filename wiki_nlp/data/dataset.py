import asyncio
import math
import multiprocessing
from collections.abc import Callable
from typing import (
    Counter,
    Iterable,
    List,
    Dict,
    OrderedDict
)

import spacy
import torch
from torch.utils.data import (
    dataset,
    get_worker_info,
    DataLoader
)
from torchtext.vocab import vocab

from wiki_nlp.data.domain import Document, Section, Paragraph
from wiki_nlp.data.service import DocumentService


class TextPreprocessor:

    def __init__(self):
        self._preprocessor = spacy.load('de_core_news_lg')

    def _tokenize(self, text: str) -> Iterable[str]:
        # Remove punctuations and numbers
        # Lemmatize and then lowercase
        return (w.lemma_.lower() for w in self._preprocessor(text) if not w.is_punct and not w.like_num)

    def tokenize_paragraph(self, par: Paragraph) -> Iterable[str]:
        return self._tokenize(par.text)

    def tokenize_section(self, sec: Section) -> Iterable[str]:
        for p in sec.paragraphs:
            for w in self.tokenize_paragraph(p):
                yield w

    def tokenize_document(self, doc: Document) -> Iterable[str]:
        excerpt_section = Section(
            id='',
            title='Kurzbeschreibung',
            position='0.0',
            paragraphs=[Paragraph(id='', text=doc.excerpt)]
        )
        doc.sections.insert(0, excerpt_section)
        for s in doc.sections:
            if s.title != 'Literatur' and s.title != 'Siehe auch' and s.title != 'Weblinks':
                for w in self.tokenize_section(s):
                    yield w


class WikiService:

    def __init__(self):
        self._preprocessor = TextPreprocessor()

    async def _get_preprocessed_documents(self, start: int, end: int) -> List[Dict[str, List[str]]]:
        documents = []
        async with DocumentService() as ds:
            for offset in range(start, end, 10):
                async for doc in ds.read_document_batch(offset=offset, limit=10):
                    documents.append({
                        'id': doc.id,
                        'text': list(self._preprocessor.tokenize_document(doc))
                    })
        return documents

    async def _get_preprocessed_sections(self, start: int, end: int) -> List[Dict[str, List[str]]]:
        sections = []
        async with DocumentService() as ds:
            for offset in range(start, end, 10):
                async for doc in ds.read_document_batch(offset=offset, limit=10):
                    for s in doc.sections:
                        sections.append({
                            'id': s.id,
                            'text': list(self._preprocessor.tokenize_section(s))
                        })
        return sections

    async def _get_preprocessed_paragraphs(self, start: int, end: int) -> List[Dict[str, List[str]]]:
        paragraphs = []
        async with DocumentService() as ds:
            for offset in range(start, end, 10):
                async for doc in ds.read_document_batch(offset=offset, limit=10):
                    for s in doc.sections:
                        for p in s.paragraphs:
                            paragraphs.append({
                                'id': p.id,
                                'text': list(self._preprocessor.tokenize_paragraph(p))
                            })
        return paragraphs

    def read_documents(self, start: int, end: int) -> List[Dict[str, List[str]]]:
        return asyncio.run(self._get_preprocessed_documents(start, end))

    def read_sections(self, start: int, end: int) -> List[Dict[str, List[str]]]:
        return asyncio.run(self._get_preprocessed_sections(start, end))

    def read_paragraphs(self, start: int, end: int) -> List[Dict[str, List[str]]]:
        return asyncio.run(self._get_preprocessed_paragraphs(start, end))


#ReaderCallback = Callable[[int, int], List[Dict[str, List[str]]]]


class WikiDataStreamer(dataset.IterableDataset):
    # A concurrent data streamer that extracts document batches
    # from the document service

    def __init__(self, start: int, end: int, reader):
        super(WikiDataStreamer, self).__init__()
        self._start = start
        self._end = end
        self._reader = reader

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = self._start
            iter_end = self._end
        else:
            # Equally distribute the document extraction among the worker processes
            per_worker = int(
                math.ceil((self._end - self._start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self._start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self._end)

        return iter(self._reader(iter_start, iter_end))


class WikiExample:
    # An example used for training
    # The example consists of its unique document id and
    # a list of words that are part of the document

    def __init__(self, id: str, text: List[str]):
        self.id = id
        self.text = text

    def __str__(self) -> str:
        return f"WikiExample=(id={self.id}, text={self.text})"

    def __repr__(self) -> str:
        return self.__str__()


class WikiDataset(dataset.Dataset):
    # A Wikipedia Dataset consists of a list of examples

    def __init__(self, examples: List[WikiExample]):
        super(WikiDataset, self).__init__()
        self._examples = examples
        self.counter = None

    def __getitem__(self, index) -> WikiExample:
        return self._examples[index]

    def __len__(self) -> int:
        return len(self._examples)

    def build_vocab(self, word_counter: Counter):
        # Discard infrequent words.
        # They do not obtain meaningful representations
        # due to their infrequency.
        # This means that they only add noise to the training set
        # that may even cause overfitting.
        self.counter = word_counter

        for example in self._examples:
            t = example.text.copy()
            example.text = []
            for w in t:
                if self.counter[w] >= 3:
                    example.text.append(w)
                else:
                    del self.counter[w]

        self.vocab = vocab(OrderedDict(self.counter.items()), min_freq=3)
        self.vocab.set_default_index(len(self.vocab))

    @staticmethod
    def from_document_streamer(
        streamer: WikiDataStreamer,
        num_workers=multiprocessing.cpu_count()
    ) -> 'WikiDataset':
        loader = DataLoader(streamer, num_workers=num_workers)
        counter = Counter()
        examples = []

        for i, doc in enumerate(loader):
            doc_id = doc['id'][0]
            examples.append(WikiExample(id=doc_id, text=[
                            d[0] for d in doc['text'] if d]))
            counter.update(examples[i].text)

        dataset = WikiDataset(examples)
        dataset.build_vocab(counter)

        return dataset

    @staticmethod
    def create_test_set(train_set: 'WikiDataset', test_data: List[WikiExample]) -> 'WikiDataset':
        test_set = WikiDataset(test_data)
        if train_set.counter is None:
            raise ValueError(
                "Cannot create a test set because the training set does not have a word counter")

        test_set.counter = train_set.counter
        test_set.vocab = train_set.vocab
        return test_set


def load_dataset(reader, start: int = 0, end: int = 2561) -> WikiDataset:
    streamer = WikiDataStreamer(start=start, end=end, reader=reader)
    return WikiDataset.from_document_streamer(streamer)


if __name__ == '__main__':
    s = WikiService()
    reader = s.read_paragraphs
    torch.save(load_dataset(reader), "paragraph_dataset")
