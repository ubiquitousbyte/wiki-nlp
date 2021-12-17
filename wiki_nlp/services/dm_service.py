from typing import Tuple, Union

import torch
from torch.optim import SGD
from torch.nn.functional import cosine_similarity

from wiki_nlp.models.train import run_loaded_dm
from wiki_nlp.models.batch_generator import BatchGenerator
from wiki_nlp.models.dm import DM
from wiki_nlp.data.domain import (
    Document,
    Section,
    Paragraph
)
from wiki_nlp.data.dataset import (
    TextPreprocessor,
    WikiDataset,
    WikiExample
)


def _load_dm_infer_model(
    model_state_path: str,
    dataset: WikiDataset,
    embedding_size: int = 100
) -> Tuple[DM, SGD, int]:
    model_state = torch.load(model_state_path)
    model = DM(embedding_dim=embedding_size, n_docs=len(
        dataset), n_words=len(dataset.vocab))
    model.load_state_dict(model_state['model_state_dict'])
    model._W.requires_grad = False
    model._Wp.requires_grad = False

    optimizer = SGD(params=model.parameters(), lr=0.1)
    optimizer.load_state_dict(model_state['optimizer_state_dict'])
    return model, optimizer, model_state['epoch']


class IDMService:

    def infer_vector(self, doc: Union[Document, Section, Paragraph]) -> torch.FloatTensor:
        pass

    def most_similar(self, vector: torch.FloatTensor, topn=10) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        pass


class DMService(IDMService):

    def __init__(self, model_state_path: str, dataset_path: str):
        self._dataset = torch.load(dataset_path)
        self._model, self._optimizer, self._epochs = _load_dm_infer_model(
            model_state_path, self._dataset)
        self._preprocessor = TextPreprocessor()

    def _create_example_from_domain(self, doc: Union[Document, Section, Paragraph]) -> WikiExample:
        example = WikiExample(id=doc.id, text=[])

        if isinstance(doc, Document):
            text = self._preprocessor.tokenize_document(doc)
        elif isinstance(doc, Section):
            text = self._preprocessor.tokenize_section(doc)
        elif isinstance(doc, Paragraph):
            text = self._preprocessor.tokenize_paragraph(doc)
        else:
            raise ValueError(
                "Document must be of type Document, Section or Paragraph")

        # Discard words that do not appear in the vocabulary
        for w in text:
            if w in self._dataset.vocab:
                example.text.append(w)
        return example

    def __getitem__(self, index) -> WikiExample:
        return self._dataset[index]

    def infer_vector(self, doc: Union[Document, Section, Paragraph]) -> torch.FloatTensor:
        example = self._create_example_from_domain(doc)
        return self._infer_vector(example)

    def _infer_vector(self, example: WikiExample) -> torch.FloatTensor:
        # Create a vector for the document
        d = torch.randn(1, self._model._D.size()[1])

        # Add the vector to the document matrix
        self._model._D.data = torch.cat((d, self._model._D))

        # Create a dataset for the batch generator
        # The dataset will hold only the document for which
        # we'd like to infer a vector.
        # However, it must also contain the vocabulary of the training set
        test_set = WikiDataset.create_test_set(self._dataset, [example])

        # Create the batch generator
        batch_generator = BatchGenerator(
            dataset=test_set,
            batch_size=128,
            ctx_size=3,
            noise_size=5,
            max_size=5,
            num_workers=1
        )
        # Start the worker
        batch_generator.start()

        # Descend on the document vector
        run_loaded_dm(
            batch_generator=batch_generator.forward(),
            model=self._model,
            optimizer=self._optimizer,
            batch_count=len(batch_generator),
            vocab_size=len(test_set.vocab),
            epochs=self._epochs
        )

        # Stop the worker
        batch_generator.stop()

        # Return the infered vector to the caller
        vec = self._model._D[0, :].detach().clone()
        # Remove the infered vector from the document matrix
        self._model._D.data = self._model._D[1:, :].data
        return vec

    def most_similar(self, vector: torch.FloatTensor, topn=10) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        D = self._model._D.data.clone().detach()
        # Compute the cosine similarity between the document matrix and the target vector
        dists = cosine_similarity(D,  vector.unsqueeze(0))
        # Extract the topn most similar documents from the document matrix
        ms = torch.topk(dists, topn)
        # Return the distances and the indices of the most similar documents
        return ms.values, ms.indices
