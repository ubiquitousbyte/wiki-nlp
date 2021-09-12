from sys import stdout
from typing import Generator 
from time import time

import torch
from torch.optim import Adam 

from wiki_nlp.models.batch_generator import BatchGenerator 
from wiki_nlp.models.dm import DM
from wiki_nlp.models.ngloss import NegativeSamplingLoss
from wiki_nlp.data.dataset import (
    load_dataset,
    document_reader,
    paragraph_reader,
    WikiDataset
)

def _run_dm(
    dataset: WikiDataset,
    batch_generator: Generator,
    batch_count: int, 
    vocab_size: int, 
    embedding_size: int = 300,
    epochs: int = 20,
    alpha: float = 0.025
):
    # This function represents the tranining loop that learns document vector representations 

    model = DM(embedding_dim=embedding_size, n_docs=len(dataset), n_words=vocab_size)
    loss_func = NegativeSamplingLoss()
    # We use Adam instead of SGD to speed up convergence rates. 
    optimizer = Adam(params=model.parameters(), lr=alpha)

    # Use the GPU whenever possible 
    #if torch.cuda.is_available():
    #    model.cuda()

    # We keep track of the best loss to pick the best weights after training 
    best_loss = float("inf")

    # Training loop 
    for epoch_idx in range(epochs):
        loss = []

        for batch_idx in range(batch_count):
            # Sample a batch from the generator 
            batch = next(batch_generator)
            #if torch.cuda.is_available():
            #    batch.cudify()
            
            # Forward pass 
            u = model.forward(batch.ctx_ids, batch.doc_ids, batch.tn_ids)

            # Cost
            J = loss_func.forward(u)
            loss.append(J.item())

            # Backward pass 
            model.zero_grad()
            J.backward()
            optimizer.step()
            _print_step(batch_idx, epoch_idx, batch_count)

        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }

        save_state(state, is_best_loss)

def save_state(state, is_best_loss: bool):
    if is_best_loss:
        torch.save(state, "state")

def _print_step(batch_idx: int, epoch_idx: int,  batch_count: int):
    step_progress = round((batch_idx + 1) / batch_count * 100)
    print("\rEpoch {:d}".format(epoch_idx + 1), end='')
    stdout.write(" - {:d}%".format(step_progress))
    stdout.flush()

if __name__ == '__main__':
    dataset = load_dataset(document_reader, end=2000) 
    batch_generator = BatchGenerator(
        dataset=dataset,
        batch_size=128,
        ctx_size=4,
        noise_size=8,
        max_size=8,
        num_workers=4
    )
    batch_generator.start()
    try:
        _run_dm(
            dataset=dataset,
            batch_generator=batch_generator.forward(),
            batch_count=len(batch_generator),
            vocab_size=len(dataset.vocab),
        )
    except KeyboardInterrupt:
        batch_generator.stop()
