from typing import (
    Dict,
    Tuple,
    List
)
from wiki_nlp.data.dataset import (
    WikiDataset,
    WikiExample
)
import torch

from wiki_nlp.models.som import SOM


class ISOMService:

    def get_map(self) -> Dict[Tuple[int, int], List[int]]:
        pass

    def winner(self, x: torch.FloatTensor) -> Tuple[int, int]:
        pass


class SOMService(ISOMService):

    def __init__(self, model_state_path: str, dataset_path: str):
        self._dataset = torch.load(dataset_path)
        model_state = torch.load(model_state_path)
        self._model = SOM(x_size=18, y_size=18, w_size=100)
        self._activation_map = model_state['activation_map']
        self._model.load_state_dict(model_state['model_state_dict'])

    def get_map(self) -> Dict[Tuple[int, int], List[int]]:
        return self._activation_map

    def winner(self, x: torch.FloatTensor) -> Tuple[int, int]:
        return self._model.winner(x)
