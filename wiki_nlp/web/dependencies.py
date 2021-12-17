
from wiki_nlp.services import (
    dm_service,
    som_service
)

from functools import lru_cache


@lru_cache
def get_dm_document_service() -> dm_service.DMService:
    return dm_service.DMService(model_state_path='document_dm_state',
                                dataset_path='document_dataset')


@lru_cache
def get_dm_paragraph_service() -> dm_service.DMService:
    return dm_service.DMService(model_state_path='paragraph_dm_state',
                                dataset_path='paragraph_dataset')


@lru_cache
def get_som_document_service() -> som_service.SOMService:
    return som_service.SOMService(model_state_path='document_som_state',
                                  dataset_path='document_dataset')


@lru_cache
def get_som_paragraph_service() -> som_service.SOMService:
    return som_service.SOMService(model_state_path='paragraph_som_state',
                                  dataset_path='paragraph_dataset')
