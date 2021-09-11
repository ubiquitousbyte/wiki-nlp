import asyncio 
import math
import re
import multiprocessing 
from collections.abc import Callable
from typing import (
    Counter,
    Iterator, 
    List,
    Dict,
    OrderedDict,
)

from torch.utils.data import (
    dataset, 
    get_worker_info,
    DataLoader
)
from torchtext.vocab import vocab

from wiki_nlp.data.domain import Document
from wiki_nlp.data.service import DocumentService

ALPHABETIC_REGEX = re.compile('(((?![\d])(\w|ä|ü|ö))+)', re.UNICODE)

def doc2text(doc: Document) -> Iterator[str]:
    for sections in doc.sections:
        for paragraph in sections.paragraphs:
             for match in ALPHABETIC_REGEX.finditer(paragraph.text.lower()):
                yield match.group()

def doc2pars(doc: Document) -> Iterator[Dict[str, List[str]]]:
    for section in doc.sections:
        for paragraph in section.paragraphs:
            words = []
            for match in ALPHABETIC_REGEX.finditer(paragraph.text.lower()):
                words.append(match.group())
            yield {'id': paragraph.id, 'text': words}

def doc2sections(doc: Document) -> Iterator[Dict[str, List[str]]]:
    for section in doc.sections:
        words = []
        for paragraph in section.paragraphs:
            for match in ALPHABETIC_REGEX.finditer(paragraph.text.lower()):
                words.append(match.group())
        yield {'id': section.id, 'text': words}

async def read_and_process_documents(start: int, end: int) -> List[Dict[str, List[str]]]:
    documents = []
    async with DocumentService() as ds:
        for offset in range(start, end, 10):
            async for doc in ds.read_document_batch(offset=offset, limit=10):
                documents.append({'id': doc.id, 'text': list(doc2text(doc))})
    return documents

async def read_and_process_paragraphs(start: int, end: int) -> List[Dict[str, List[str]]]:
    paragraphs = []
    async with DocumentService() as ds:
        for offset in range(start, end, 10):
            async for doc in ds.read_document_batch(offset=offset, limit=10):
                for par in doc2pars(doc):
                    paragraphs.append(par)
    return paragraphs 

async def read_and_process_sections(start: int, end: int) -> List[Dict[str, List[str]]]:
    sections = []
    async with DocumentService() as ds:
        for offset in range(start, end, 10):
            async for doc in ds.read_document_batch(offset=offset, limit=10):
                for sec in doc2sections(doc):
                    sections.append(sec)
    return sections 

# A function of type (in Scala syntax) f: (Int, Int) => List[Map[str, List[str]]]
ReaderCallback = Callable[[int, int], List[Dict[str, List[str]]]]

# A reader callback that extracts all documents from the corpus
def document_reader(start: int, end: int) -> List[Dict[str, List[str]]]:
    return asyncio.run(read_and_process_documents(start, end))

# A reader callback that extracts all sections from the corpus
def section_reader(start: int, end: int) -> List[Dict[str, List[str]]]:
    return asyncio.run(read_and_process_sections(start, end))

# A reader callback that extracts all paragraphs from the corpus 
def paragraph_reader(start: int, end: int) -> List[Dict[str, List[str]]]:
    return asyncio.run(read_and_process_paragraphs(start, end))

class WikiDataStreamer(dataset.IterableDataset):
    # A concurrent data streamer that extracts document batches 
    # from the document service 

    def __init__(self, start: int, end: int, reader: ReaderCallback):
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
            per_worker = int(math.ceil((self._end - self._start) / float(worker_info.num_workers)))
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
    # A Wikipedia Dataset consists of a map of examples, and a vocabulary 
    # constructed from those examples 
    def __init__(self, streamer: WikiDataStreamer, num_workers: int):
        super(WikiDataset, self).__init__()
        loader = DataLoader(streamer, num_workers=num_workers)
        self.counter = Counter()
        self._examples = []

        for i, doc in enumerate(loader):
            doc_id = doc['id'][0]
            self._examples.append(WikiExample(id=doc_id, text=[d[0] for d in doc['text'] if d]))
            self.counter.update(self._examples[i].text)

        self.vocab = vocab(OrderedDict(self.counter.items()), min_freq=1)

    def __getitem__(self, i):
        return self._examples[i]

    def __len__(self) -> int: 
        return len(self._examples)
        
def load_dataset(reader: ReaderCallback, start: int = 0, end: int = 2000) -> WikiDataset:
    streamer = WikiDataStreamer(start=start, end=end, reader=reader)
    return WikiDataset(streamer, num_workers=multiprocessing.cpu_count())