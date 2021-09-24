from sys import stdout
from typing import (
    Generator, 
    Optional
)

import torch
from torch.optim import SGD

from wiki_nlp.models.batch_generator import BatchGenerator 
from wiki_nlp.models.dm import DM
from wiki_nlp.models.ngloss import NegativeSamplingLoss
from wiki_nlp.data.dataset import (
    WikiDataset, 
    WikiExample,
)

def run_loaded_dm(
    batch_generator: Generator,
    model: DM, 
    optimizer: SGD, 
    batch_count: int, 
    vocab_size: int, 
    epochs: int = 140, 
    state_path: Optional[str] = None
):
    if torch.cuda.is_available():
        model.cuda()

    loss_func = NegativeSamplingLoss()

    # We keep track of the best loss to pick the best weights after training 
    best_loss = float("inf")

    # Training loop 
    for epoch_idx in range(0, epochs):
        loss = []

        for batch_idx in range(batch_count):
            # Sample a batch from the generator 
            batch = next(batch_generator)
            if torch.cuda.is_available():
                batch.cudify()
            
            # Forward pass 
            x = model.forward(batch.ctx_ids, batch.doc_ids, batch.tn_ids)
            # Cost
            J = loss_func.forward(x)
            loss.append(J.item())

            # Backward pass 
            model.zero_grad()
            J.backward()
            # Don't backpropagate through the sentinel word vector
            if model._W.grad is not None:
                model._W.grad.data[vocab_size, :].fill_(0)
           
            optimizer.step()
            _print_step(batch_idx, epoch_idx, batch_count)

        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)
        print('\nLoss', loss)

        if state_path is not None:
            state = {
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict()
            }
            save_state(state, is_best_loss, state_path)

def save_state(state, is_best_loss: bool, path: str):
    if is_best_loss:
        torch.save(state, path)

def _print_step(batch_idx: int, epoch_idx: int,  batch_count: int):
    step_progress = round((batch_idx + 1) / batch_count * 100)
    print("\rEpoch {:d}".format(epoch_idx + 1), end='')
    stdout.write(" - {:d}%".format(step_progress))
    stdout.flush()

def _run_dm(
    dataset: WikiDataset,
    batch_generator: Generator,
    batch_count: int, 
    vocab_size: int, 
    state_path: str,
    embedding_size: int = 100,
    epochs: int = 140,
    alpha: float = 0.1, 
):
    model = DM(embedding_dim=embedding_size, n_docs=len(dataset), n_words=vocab_size)
    optimizer = SGD(params=model.parameters(), lr=alpha)
    run_loaded_dm(
        batch_generator, 
        model, 
        optimizer, 
        batch_count, 
        epochs, 
        state_path
    )

if __name__ == '__main__':
    dataset = torch.load("paragraph_dataset")
    batch_generator = BatchGenerator(
        dataset=dataset,
        batch_size=128,
        ctx_size=3,
        noise_size=5,
        max_size=5,
        num_workers=1
    )
    batch_generator.start()
    try:
        _run_dm(
            dataset=dataset,
            batch_generator=batch_generator.forward(),
            batch_count=len(batch_generator),
            vocab_size=len(dataset.vocab),
            state_path="paragraph_dm_state"
        )
    except KeyboardInterrupt:
        batch_generator.stop()
