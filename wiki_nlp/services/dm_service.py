from typing import Tuple 

import torch
from torch.optim import SGD 
from torch.nn.functional import cosine_similarity

from wiki_nlp.models.train import run_loaded_dm
from wiki_nlp.models.batch_generator import BatchGenerator
from wiki_nlp.models.dm import DM 
from wiki_nlp.data.dataset import (
    WikiDataset, 
    WikiExample
)

def load_dm_infer_model(
    model_state_path: str, 
    dataset: WikiDataset, 
    embedding_size: int = 100
) -> Tuple[DM, SGD, int]:
    model_state = torch.load(model_state_path)
    model = DM(embedding_dim=embedding_size, n_docs=len(dataset), n_words=len(dataset.vocab))
    model.load_state_dict(model_state['model_state_dict'])
    model._W.requires_grad = False 
    model._Wp.requires_grad = False 

    optimizer = SGD(params=model.parameters(), lr=0.1)
    optimizer.load_state_dict(model_state['optimizer_state_dict'])
    return model, optimizer, model_state['epoch']

class DMService:

    def __init__(self, model_state_path: str, dataset_path: str):
        self._dataset = torch.load(dataset_path)
        self._model, self._optimizer, self._epochs = load_dm_infer_model(model_state_path, self._dataset)

    def infer_vector(self, document: WikiExample) -> torch.FloatTensor:
        # Create a vector for the document 
        d = torch.randn(1, self._model._D.size()[1])

        # Add the vector to the document matrix
        self._model._D.data = torch.cat((d, self._model._D))

        # Create a dataset for the batch generator
        # The dataset will hold only the document for which
        # we'd like to infer a vector.
        # However, it must also contain the vocabulary of the training set
        test_set = WikiDataset.create_test_set(self._dataset, [document])

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

    def most_similar(self, vector: torch.FloatTensor, topn=10):
        D = self._model._D.data.clone().detach()
        dists = cosine_similarity(D,  vector.unsqueeze(0))
        ms = torch.topk(dists, topn)
        return ms.values, ms.indices
        
if __name__ == '__main__':
    dm_service = DMService(
        model_state_path="document_dm_state",
        dataset_path="document_dataset"
    )
    vec = dm_service.infer_vector(dm_service._dataset[1400])
    dists, indices = dm_service.most_similar(vec, topn=10)
    print(dm_service._dataset[1400])
    for i, dist in enumerate(dists):
        print(f"{dist.item()}\n {dm_service._dataset[indices[i].item()]}")